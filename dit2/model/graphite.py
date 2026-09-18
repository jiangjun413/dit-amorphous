import math
import logging
import torch
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.nn import Gate
from torch_scatter import scatter
from dit2.model.embeddings import (GaussianBasis, ScalarCondEmbed, BesselBasisRB,
                                   EqGate, RegionEmbed, MethodEmbed, RDFEmbed,
                                   _make_bn, _make_rmlp,
                                   _i0_mul0e, _inject_scalar)
from dit2.model.convolutions import NequIP_MultiConv, _build_irh_blocks

# ============================================================
# Graphite-compatible NequIP (fully inline, no graphite import)
# Exact same parameter names/shapes as graphite for torch.load compat
# ============================================================
def _bessel_basis(x, start=0.0, end=1.0, num_basis=8, eps=1e-5):
    """Bessel radial basis expansion (matches graphite.nn.basis.bessel)."""
    x = x[..., None] - start + eps
    c = end - start
    n = torch.arange(1, num_basis + 1, dtype=x.dtype, device=x.device)
    return ((2 / c) ** 0.5) * torch.sin(n * math.pi * x / c) / x

def _tp_path_exists(irreps_in1, irreps_in2, ir_out):
    """Check if a tensor product path exists."""
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)
    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False

class _GraphiteInteraction(nn.Module):
    """Equivariant Interaction layer — exact replica of graphite's
    graphite.nn.conv.e3nn_nequip.Interaction.

    Uses sparse TensorProduct with 'uvu' mode instructions,
    FCTP self-connection, FCTP pre/post linears with node_attr,
    and SkipInit alpha gating.

    Parameter names match graphite exactly: sc, lin1, conv, lin2, mlp, alpha
    """
    def __init__(self, irreps_in, irreps_node, irreps_edge, irreps_out,
                 radial_neurons=(16, 64), num_neighbors=1):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_neighbors = num_neighbors

        # Build sparse TP instructions ('uvu' mode)
        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_edge) in enumerate(self.irreps_edge):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvu', True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()
        instructions = [(i1, i2, p[io], mode, tr)
                        for i1, i2, io, mode, tr in instructions]

        # Modules — names match graphite exactly
        self.sc = o3.FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_node, self.irreps_out)
        self.lin1 = o3.FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_node, self.irreps_in)
        self.conv = o3.TensorProduct(
            self.irreps_in, self.irreps_edge, irreps_mid,
            instructions, internal_weights=False, shared_weights=False)
        self.lin2 = o3.FullyConnectedTensorProduct(
            irreps_mid, self.irreps_node, self.irreps_out)

        # Radial MLP (FullyConnectedNet compatible with all e3nn versions)
        try:
            from e3nn.nn import FullyConnectedNet
            self.mlp = FullyConnectedNet(
                list(radial_neurons) + [self.conv.weight_numel],
                torch.nn.functional.silu)
        except ImportError:
            # e3nn >= 0.5 may not have FullyConnectedNet; build manually
            dims = list(radial_neurons) + [self.conv.weight_numel]
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(nn.SiLU())
            self.mlp = nn.Sequential(*layers)

        # SkipInit alpha gate
        self.alpha = o3.FullyConnectedTensorProduct(
            irreps_mid, self.irreps_node, "0e")
        with torch.no_grad():
            self.alpha.weight.zero_()

    def forward(self, x, node_attr, edge_index, edge_attr, edge_len_emb):
        i, j = edge_index
        num_nodes = x.size(0)

        node_self = self.sc(x, node_attr)
        h = self.lin1(x, node_attr)
        edge_feat = self.conv(h[i], edge_attr,
                              weight=self.mlp(edge_len_emb))
        h = scatter(edge_feat, j, dim=0,
                    dim_size=num_nodes).div(self.num_neighbors ** 0.5)
        h_conv = self.lin2(h, node_attr)

        alpha = self.alpha(h, node_attr)
        m = self.sc.output_mask
        alpha = (1 - m) + alpha * m
        return node_self + alpha * h_conv

class _GraphiteCompose(nn.Module):
    """Sequential composition of two modules (matches graphite's Compose).
    Parameter name: .first and .second"""
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, *args):
        return self.second(self.first(*args))

class _GraphiteGaussianBasisEmbedding(nn.Module):
    """Gaussian basis embedding (matches graphite's GaussianBasisEmbedding).
    Parameter names: means, sigmas, layer1, layer2"""
    def __init__(self, num_basis=12, embedding_dim=32, min_sigma=0.1,
                 learn_means=False, learn_sigmas=False,
                 min_value=0, max_value=1):
        super().__init__()
        means = torch.linspace(min_value, max_value, num_basis)
        if learn_means:
            self.means = nn.Parameter(means)
        else:
            self.register_buffer('means', means)
        # Width tied to the ACTUAL center spacing.  The legacy expression
        # `1.0 / (num_basis - 1)` assumed a [0, 1] value range; for the
        # energy range [-15, 0] (spacing 1.0 eV) it evaluated to 0.067 so
        # min_sigma=0.3 always won — a spiky comb whose activation dips to
        # exp(-0.5·(0.5/0.3)²) ≈ 0.25 between centers, making the embedding
        # response non-uniform in the target value (measured: per-0.3 eV
        # embedding shifts varying 0.21–0.74 along a 1.2 eV sweep).
        # `sigmas` is a persisted buffer, so checkpoints (and warm starts,
        # which copy it by name+shape) keep their trained-with widths —
        # only freshly built models get the smooth basis.
        spacing = (float(max_value) - float(min_value)) / max(num_basis - 1, 1)
        sigmas = torch.ones_like(means) * max(min_sigma, spacing)
        if learn_sigmas:
            self.sigmas = nn.Parameter(sigmas)
        else:
            self.register_buffer('sigmas', sigmas)
        hidden_dim = max(embedding_dim * 2, num_basis)
        self.layer1 = nn.Linear(num_basis, hidden_dim)
        self.activation = nn.Softplus()
        self.layer2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        basis = torch.exp(
            -0.5 * ((x.expand(-1, self.means.shape[0]) - self.means)
                     / self.sigmas) ** 2)
        return self.layer2(self.activation(self.layer1(basis)))

class NequIP_EnergyEmbed(nn.Module):
    """NequIP with energy conditioning — graphite-style architecture.

    Uses graphite's Interaction layer (sparse TP with 'uvu' mode,
    FCTP self-connection, SkipInit alpha) with energy-conditioned
    additive embedding at each layer.

    Fully self-contained — no graphite import needed.

    Parameter structure:
      init_embed.embed_node_x.weight, init_embed.embed_node_z.weight
      interactions.N.first.{sc,lin1,conv,lin2,mlp,alpha}.*
      interactions.N.second.*  (Gate)
      out.weight
      e_embed.{means,sigmas,layer1,layer2}.*
      e_projection.{0,2}.*
    """
    def __init__(self, init_embed, irreps_node_x='8x0e',
                 irreps_node_z='8x0e',
                 irreps_hidden='64x0e + 32x1e',
                 irreps_edge='4x0e + 4x1e + 2x2e',
                 irreps_out='1x1e', num_convs=3,
                 radial_neurons=(16, 64), num_neighbors=12,
                 cond_min=-15.0, cond_max=0.0,
                 cond_num_basis=16, cond_min_sigma=0.3,
                 n_regions=0,
                 # Conditioning v2: energy keeps its dedicated e_embed; these
                 # add a categorical method tag, EXTRA scalar conditions
                 # (temperature / delta_e_min — never energy), and an RDF
                 # encoder, all injected into the same l=0 channels.
                 n_methods=0, extra_scalar_conditions=None, rdf_config=None,
                 # dit2 minimal-conditioning: use_energy=False builds NO
                 # absolute-energy head at all (e_embed/e_projection are None
                 # and forward ignores the energy argument).  Absolute energy
                 # is redundant with composition + delta_e_min (E = E_floor(comp)
                 # + ΔE), and its coarse basis (σ 0.3 eV/atom) only added a
                 # noisy second encoding of what the sharp ΔE basis already
                 # carries.  Callers convert a user-facing target_energy to
                 # ΔE at generation time instead.
                 use_energy=True):
        super().__init__()
        self.init_embed = init_embed
        # n_regions > 0 enables per-atom region conditioning for interface
        # models (see RegionEmbed).  Persist it so generation code can
        # tell a region-aware checkpoint apart from a bulk one.
        self.n_regions = int(n_regions)
        self.irreps_node_x = o3.Irreps(irreps_node_x)
        self.irreps_node_z = o3.Irreps(irreps_node_z)
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self.num_convs = num_convs

        act_scalars = {1: nn.functional.silu, -1: torch.tanh}
        act_gates = {1: torch.sigmoid, -1: torch.tanh}

        irreps = self.irreps_node_x
        self.interactions = nn.ModuleList()
        for _ in range(num_convs):
            irreps_scalars = o3.Irreps([
                (m, ir) for m, ir in self.irreps_hidden
                if ir.l == 0 and _tp_path_exists(irreps, self.irreps_edge, ir)])
            irreps_gated = o3.Irreps([
                (m, ir) for m, ir in self.irreps_hidden
                if ir.l > 0 and _tp_path_exists(irreps, self.irreps_edge, ir)])

            if irreps_gated.dim > 0:
                if _tp_path_exists(irreps_node_z, self.irreps_edge, "0e"):
                    ir = "0e"
                elif _tp_path_exists(irreps_node_z, self.irreps_edge, "0o"):
                    ir = "0o"
                else:
                    raise ValueError(
                        f"Cannot produce gates for irreps_gated={irreps_gated}")
            else:
                ir = None
            irreps_gates = o3.Irreps(
                [(mul, ir) for mul, _ in irreps_gated]).simplify()

            gate = Gate(
                irreps_scalars,
                [act_scalars[ir.p] for _, ir in irreps_scalars],
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],
                irreps_gated)

            conv = _GraphiteInteraction(
                irreps_in=irreps, irreps_node=self.irreps_node_z,
                irreps_edge=self.irreps_edge, irreps_out=gate.irreps_in,
                radial_neurons=radial_neurons, num_neighbors=num_neighbors)

            irreps = gate.irreps_out
            self.interactions.append(_GraphiteCompose(conv, gate))

        self.out = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps, irreps_in2=self.irreps_node_z,
            irreps_out=self.irreps_out)

        # M1: derive embedding width from the irreps' actual scalar (l=0)
        # multiplicity, not from a fragile string-split.  The legacy
        # ``int(str(irreps).split("x")[0])`` form took the leading number
        # of the string representation — correct for the default
        # ``"64x0e + 32x1e"`` (gives 64) but wrong for any future
        # irreps_hidden that doesn't start with l=0 or that mixes
        # scalars across multiple blocks (e.g. ``"32x0e + 32x1e + 32x0e"``
        # would give 32, missing half the scalar capacity).
        size_embed = sum(int(mul) for mul, ir in irreps if ir.l == 0)
        if size_embed <= 0:
            raise ValueError(
                f"NequIP_EnergyEmbed: irreps after gating must contain at "
                f"least one l=0 multiplet, got {irreps!r}.  The energy/"
                f"conditioning embedding has nowhere to project into.")
        # ── FIX 1: project conditioning only into scalar (l=0) channels ──
        # Injecting non-equivariant values into l>0 channels corrupts the
        # angular information the model learns for bond angles/orientations,
        # causing noisy RDFs.  We project into scalar_dim only and pad l>0
        # slots with zeros — identical to what DM2/graphite does implicitly.
        scalar_dim = sum(mul * (2 * ir.l + 1)
                         for mul, ir in irreps if ir.l == 0)
        self._scalar_dim = scalar_dim
        self._irreps_full_dim = irreps.dim
        self.use_energy = bool(use_energy)
        if self.use_energy:
            self.e_embed = _GraphiteGaussianBasisEmbedding(
                embedding_dim=size_embed, num_basis=cond_num_basis,
                min_value=cond_min, max_value=cond_max,
                min_sigma=cond_min_sigma)
            e_embed_dim = self.e_embed.layer2.out_features
            self.e_projection = nn.Sequential(
                nn.Linear(e_embed_dim, e_embed_dim),
                nn.SiLU(),
                nn.Linear(e_embed_dim, scalar_dim))  # scalar channels only
        else:
            self.e_embed = None
            self.e_projection = None

        # Optional per-atom region conditioning (interface models).  Emits
        # a scalar-channel bias per atom, added at the same injection
        # point as the energy embedding.  Absent (n_regions==0) → the
        # model is a pure bulk denoiser and forward ignores data.region.
        if self.n_regions > 0:
            self.region_embed = RegionEmbed(self.n_regions, scalar_dim)

        # ── Conditioning v2 heads (all emit a scalar_dim bias) ──
        self.n_methods = int(n_methods or 0)
        if self.n_methods > 0:
            self.method_embed = MethodEmbed(self.n_methods, scalar_dim)
        self.extra_cond_names = (sorted(extra_scalar_conditions.keys())
                                 if extra_scalar_conditions else [])
        if self.extra_cond_names:
            # New scalars are zero-init so warm start from an energy-only
            # graphite checkpoint reproduces the baseline exactly.
            self.extra_sc = ScalarCondEmbed(extra_scalar_conditions, scalar_dim,
                                            zero_init=True)
        if rdf_config is not None:
            self.rdf_embed = RDFEmbed(
                n_bins=rdf_config['n_bins'], dim=scalar_dim,
                n_pairs=rdf_config['n_pairs'],
                total_channels=rdf_config['total_channels'],
                use_partial=rdf_config.get('use_partial', True))

    def forward(self, data, energy):
        data = self.init_embed(data)
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        h_node_x = data.h_node_x
        h_node_z = data.h_node_z
        h_edge = data.h_edge

        batch = getattr(data, 'batch', None)
        if batch is None:
            # PyG Data returns None (not AttributeError) for absent batch,
            # so a getattr default never fires on single-graph inputs.
            batch = torch.zeros(h_node_x.size(0),
                                device=h_node_x.device, dtype=torch.long)

        # ── FIX 1 (cont.): build scalar-only conditioning, zero-pad l>0 slots ──
        _ng = int(batch.max().item()) + 1 if batch.numel() else 1
        if getattr(self, 'e_embed', None) is not None:
            h_node_e_s = self.e_projection(self.e_embed(energy)[batch])  # [N, scalar_dim]
            # Optional energy presence mask (Conditioning v2): 0 → generate
            # UNconditioned on energy.  Masking the output keeps e_embed's
            # parameters in the graph (DDP-safe under CondPresenceDropout).
            _ep = getattr(data, 'energy_present', None)
            if _ep is not None:
                h_node_e_s = h_node_e_s * _ep.view(-1)[:_ng][batch].unsqueeze(-1).to(h_node_e_s.dtype)
        else:
            # dit2 minimal conditioning (use_energy=False): no absolute-energy
            # head — conditioning starts from zero and is carried entirely by
            # the delta_e_min / method / RDF heads below.
            h_node_e_s = torch.zeros(
                h_node_x.size(0), self._scalar_dim,
                device=h_node_x.device, dtype=h_node_x.dtype)
        # Add the per-atom region bias into the same scalar channels when
        # this is a region-aware model and the input carries region ids.
        region = getattr(data, 'region', None)
        if getattr(self, 'n_regions', 0) > 0 and hasattr(self, 'region_embed') \
                and region is not None:
            h_node_e_s = h_node_e_s + self.region_embed(region)

        # ── Conditioning v2 injection (same l=0 channels) ──
        ng = _ng
        if getattr(self, 'n_methods', 0) > 0 and hasattr(self, 'method_embed'):
            m = getattr(data, 'method', None)
            if m is None:
                m = torch.zeros(ng, dtype=torch.long, device=h_node_e_s.device)
            mout = self.method_embed(m.view(-1).long())
            mp = getattr(data, 'method_present', None)
            if mp is not None:
                mout = mout * mp.view(-1)[:ng].view(ng, 1).to(mout.dtype)
            h_node_e_s = h_node_e_s + mout[batch]
        if getattr(self, 'extra_cond_names', None):
            sc, pres = {}, {}
            for name in self.extra_cond_names:
                val = getattr(data, name, None)
                if val is not None:
                    if val.dim() == 0:
                        sc[name] = val.unsqueeze(0)
                    elif val.numel() == ng:
                        sc[name] = val.view(ng)
                    else:
                        sc[name] = val.view(ng, -1)[:, 0]
                    p = getattr(data, f'{name}_present', None)
                    if p is not None:
                        pres[name] = p.view(-1)[:ng]
            if sc:
                h_node_e_s = h_node_e_s + self.extra_sc(
                    sc, batch, ng=ng, present=pres if pres else None)
        if hasattr(self, 'rdf_embed'):
            h_node_e_s = h_node_e_s + self.rdf_embed(data, batch, ng)

        pad_dim = self._irreps_full_dim - self._scalar_dim
        if pad_dim > 0:
            h_node_e = torch.cat(
                [h_node_e_s,
                 torch.zeros(h_node_e_s.size(0), pad_dim,
                             device=h_node_e_s.device, dtype=h_node_e_s.dtype)],
                dim=-1)
        else:
            h_node_e = h_node_e_s

        edge_sh = o3.spherical_harmonics(
            self.irreps_edge, edge_attr, normalize=True,
            normalization='component')
        for layer in self.interactions:
            h_node_x = layer(h_node_x, h_node_z, edge_index, edge_sh,
                             h_edge)
            h_node_x = h_node_x + h_node_e

        return self.out(h_node_x, h_node_z)

class _GraphiteInitialEmbedding(nn.Module):
    """InitialEmbedding with inline bessel basis (no graphite dependency).

    Embeds atomic numbers Z directly through a periodic-table-wide table
    of size 119 (indices 0..118; Z=0 is padding_idx and stays zero).
    `element_latent_dim` (default 8) sets the embedding output width — same
    value used by `irreps_node_x` / `irreps_node_z` in the downstream model.
    """
    # Periodic-table vocab size shared with NequIP_MultiConv.
    PERIODIC_TABLE_SIZE = 119

    def __init__(self, cutoff, num_basis=16, element_latent_dim=8):
        super().__init__()
        self.element_latent_dim = int(element_latent_dim)
        self.embed_node_x = nn.Embedding(
            self.PERIODIC_TABLE_SIZE, self.element_latent_dim, padding_idx=0)
        self.embed_node_z = nn.Embedding(
            self.PERIODIC_TABLE_SIZE, self.element_latent_dim, padding_idx=0)
        self._cutoff = cutoff
        self._num_basis = num_basis

    def forward(self, data):
        data.h_node_x = self.embed_node_x(data.x)
        data.h_node_z = self.embed_node_z(data.x)
        data.h_edge = _bessel_basis(
            data.edge_attr.norm(dim=-1),
            start=0.0, end=self._cutoff, num_basis=self._num_basis)
        return data

class GraphiteModelAdapter(nn.Module):
    """Wraps NequIP_EnergyEmbed to match dit.py forward(data) interface.
    Reads energy from data.energy.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, data):
        energy = getattr(data, 'energy', None)
        if energy is None:
            energy = torch.zeros(1, device=data.pos.device)
        return self.model(data, energy)

def _build_graphite_model(cfg, scc, logger):
    """Build inline NequIP_EnergyEmbed (no graphite import needed)."""
    init_embed = _GraphiteInitialEmbedding(
        cutoff=cfg['cutoff'],
        element_latent_dim=cfg.get('element_latent_dim', 8))
    model = NequIP_EnergyEmbed(
        init_embed=init_embed,
        irreps_node_x=cfg['irreps_node_x'],
        irreps_node_z=cfg['irreps_node_z'],
        irreps_hidden=cfg['irreps_hidden'],
        irreps_edge=cfg['irreps_edge'],
        irreps_out=cfg.get('irreps_out', '1x1e'),
        num_convs=cfg['num_convs'],
        radial_neurons=cfg['radial_neurons'],
        num_neighbors=cfg['num_neighbors'],
        n_regions=int(cfg.get('n_regions', 0)),
        # Conditioning v2: energy stays on e_embed; extra scalars are the
        # non-energy conditions from scc (temperature / delta_e_min).
        n_methods=int(cfg.get('n_methods', 0)),
        extra_scalar_conditions={n: scc[n] for n in (scc or {}) if n != 'energy'} or None,
        rdf_config=cfg.get('rdf_config'),
        # dit2: the absolute-energy head exists only when 'energy' is an
        # explicit scalar condition; the minimal recipe (delta_e_min +
        # method + partial RDF) builds no e_embed at all.
        use_energy=('energy' in (scc or {})))
    v2 = []
    if int(cfg.get('n_regions', 0)) > 0: v2.append(f"n_regions={cfg.get('n_regions')}")
    if int(cfg.get('n_methods', 0)) > 0: v2.append(f"n_methods={cfg.get('n_methods')}")
    if cfg.get('rdf_config'): v2.append("rdf")
    extra = [n for n in (scc or {}) if n != 'energy']
    if extra: v2.append("scalars=" + "+".join(extra))
    logger.info("  Built NequIP_EnergyEmbed (graphite-style, inline)"
                + (f" [{', '.join(v2)}]" if v2 else ""))
    return GraphiteModelAdapter(model)
