import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.nn import Gate
from torch_scatter import scatter

class InitialEmbedding(nn.Module):
    """Standalone initial embedding — no graphite dependency.
    Kept for backward-compatible torch.load of old models.
    """
    def __init__(self, num_species: int, cutoff: float):
        super().__init__()
        self.embed_node_x = nn.Embedding(num_species, 8)
        self.embed_node_z = nn.Embedding(num_species, 8)
        self._cutoff = cutoff
        self.register_buffer('_bessel_freqs',
            torch.arange(1, 17).float() * math.pi / cutoff)
        # `_bessel_freqs` already encodes nπ/cutoff, so forward must not
        # divide by cutoff again (the legacy code did, yielding
        # sin(nπd/c²) — a basis 1/c as resolving as intended).  Old
        # whole-module pickles lack this attribute after unpickling
        # (__init__ is not re-run), so they keep the exact legacy math
        # their downstream weights were trained against.
        self._freq_fixed = True

    def _bessel(self, d):
        d = d.clamp(min=1e-8)
        arg = self._bessel_freqs * d.unsqueeze(-1)
        if not getattr(self, '_freq_fixed', False):
            arg = arg / self._cutoff   # legacy pre-fix checkpoints
        b = arg.sin() * math.sqrt(2.0 / self._cutoff) / d.unsqueeze(-1)
        x = d / self._cutoff
        env = (1 - 6*x**5 + 15*x**4 - 10*x**3).clamp(min=0.0)
        return b * env.unsqueeze(-1)

    def forward(self, data):
        data.h_node_x = self.embed_node_x(data.x)
        data.h_node_z = self.embed_node_z(data.x)
        data.h_edge = self._bessel(data.edge_attr.norm(dim=-1))
        return data

# ============================================================
# Multi-Property Conditioning (energy, density, etc.)
# ============================================================
class GaussianBasis(nn.Module):
    """Gaussian radial basis expansion for scalar conditioning."""
    def __init__(self, vmin, vmax, n_basis=32, log_scale=False):
        super().__init__()
        self.log_scale = log_scale
        c = torch.linspace(
            math.log(max(vmin, 1e-10)) if log_scale else vmin,
            math.log(vmax) if log_scale else vmax, n_basis)
        self.register_buffer('centers', c)
        # Guard against a degenerate vmin==vmax range (or n_basis==1), which
        # would make width 0 and emit NaN from the exp() in forward.
        _w = (c[-1] - c[0]) / max(n_basis - 1, 1)
        if not (_w > 0):
            _w = 1.0
        self.register_buffer('width', _w * torch.ones(1))

    def forward(self, x):
        if self.log_scale:
            x = torch.log(x.clamp(min=1e-10))
        return torch.exp(-0.5 * ((x.unsqueeze(-1) - self.centers) / self.width) ** 2)

class ScalarCondEmbed(nn.Module):
    """Multi-property conditioning: energy, density, pressure, etc.
    Each condition: Gaussian basis + MLP → scalar embedding, summed.
    """
    def __init__(self, configs, dim, hid=64, zero_init=False):
        super().__init__()
        self.names = sorted(configs.keys())
        self.dim = dim
        self.basis = nn.ModuleDict()
        self.mlps = nn.ModuleDict()
        for n in self.names:
            c = configs[n]
            nb = c.get('n_basis', 32)
            self.basis[n] = GaussianBasis(
                c['vmin'], c['vmax'], nb, c.get('log_scale', False))
            mlp = nn.Sequential(
                nn.Linear(nb, hid), nn.SiLU(), nn.Linear(hid, dim))
            # Conditioning v2: zero-init the final Linear so a freshly-added
            # scalar-conditioning head contributes nothing at step 0 — a
            # warm-started model then reproduces its baseline exactly, and the
            # head ramps in during fine-tuning.  Existing checkpoints overwrite
            # this at load, so they are unaffected.
            if zero_init:
                nn.init.zeros_(mlp[-1].weight)
                nn.init.zeros_(mlp[-1].bias)
            self.mlps[n] = mlp

    def forward(self, conds, batch, ng=None, present=None):
        if ng is None:
            ng = batch.max().item() + 1
        # Start from the same dtype as the mlp output so autocast (bf16/fp16)
        # doesn't get silently promoted to fp32 by an fp32 zeros() initializer.
        _dtype = next(self.mlps[self.names[0]].parameters()).dtype
        e = torch.zeros(ng, self.dim, device=batch.device, dtype=_dtype)
        for n in self.names:
            if n in conds and conds[n] is not None:
                out = self.mlps[n](self.basis[n](conds[n]))
                # Optional per-graph presence mask (Conditioning v2): masking
                # the OUTPUT keeps every parameter in the autograd graph
                # (DDP-safe) while a masked condition contributes exactly
                # zero — the "not conditioned on this axis" mode that
                # CondPresenceDropout exposes during training.
                if present is not None and n in present and present[n] is not None:
                    out = out * present[n].view(ng, 1).to(out.dtype)
                e = e + out
        return e[batch]

class RegionEmbed(nn.Module):
    """Per-atom region / phase embedding for interface conditioning.

    An interface structure is a two-(or more-)phase system: each atom
    belongs to a discrete region (e.g. 0 = substrate, 1 = film).  Bulk
    conditioning is a single global scalar (energy / density) shared by
    every atom, so it cannot express that the two sides of an interface
    have different local order.  This module maps a per-atom integer
    region id → a learned scalar-channel vector that is added into the
    l=0 (invariant) channels alongside the energy embedding, exactly the
    same injection point.  Adding into scalars only keeps the model
    SO(3)-equivariant (a per-atom scalar bias is invariant).

    `padding_idx=0` is NOT used: region 0 is a real phase, not padding.
    The module is only constructed for region-aware models; a plain bulk
    checkpoint has no `region_embed`, so loading it is unaffected.
    """
    def __init__(self, n_regions, dim):
        super().__init__()
        self.n_regions = int(n_regions)
        self.embed = nn.Embedding(self.n_regions, dim)
        # Start at zero so a freshly region-enabled model reproduces the
        # bulk model's output on step 0 (the region bias ramps in during
        # training / fine-tuning rather than perturbing a warm start).
        nn.init.zeros_(self.embed.weight)

    def forward(self, region):
        return self.embed(region)


class MethodEmbed(nn.Module):
    """Per-structure categorical simulation-method embedding.

    A learned additive scalar-channel bias per method id (0..n_methods-1):
    PBE / classical / MLP / … .  Separating energies by the method that
    produced them lets the model interpret a target energy on the right scale
    instead of conflating (e.g.) a PBE and a classical-potential energy.

    Mirrors `RegionEmbed` but is PER-GRAPH, not per-atom — so callers index the
    embedding by the per-graph method id and broadcast with `[batch]`.
    Zero-initialised so a freshly method-enabled model reproduces the baseline
    (method id whose row is all-zero is a genuine no-op at warm start).
    """
    def __init__(self, n_methods, dim):
        super().__init__()
        self.n_methods = int(n_methods)
        self.embed = nn.Embedding(self.n_methods, dim)
        nn.init.zeros_(self.embed.weight)

    def forward(self, method):
        return self.embed(method)


class RDFEmbed(nn.Module):
    """Encode RDF conditioning channels into an additive scalar-channel bias.

    Handles a family of INDEPENDENTLY-OPTIONAL channels — partial g_AB(r)
    (`n_pairs x n_bins`) plus any of four Faber-Ziman totals (number / X-ray /
    electron / neutron, each `n_bins`).  Each channel has its own encoder whose
    final Linear is zero-initialised (warm-start transparent).

    DDP-safety: every encoder is ALWAYS executed on a fixed-shape tensor
    (zeros when a channel is absent), so every parameter participates on every
    rank; a per-graph presence mask multiplies each encoder output before the
    sum, so a masked channel contributes exactly zero without a None branch.
    The partial encoder is shared across the `n_pairs` pair-channels (a
    per-pair learned id-bias distinguishes them) and summed with a per-graph
    per-pair presence mask (absent pairs are zero rows).

    `total_channels` is the ordered list of active total names (subset of
    number/xray/electron/neutron); `use_partial` toggles the partial encoder.
    """
    def __init__(self, n_bins, dim, n_pairs, total_channels,
                 use_partial=True, hid=64):
        super().__init__()
        self.n_bins = int(n_bins)
        self.dim = int(dim)
        self.n_pairs = int(n_pairs)
        self.total_channels = list(total_channels)
        self.use_partial = bool(use_partial)

        def _enc():
            m = nn.Sequential(nn.Linear(self.n_bins, hid), nn.SiLU(),
                              nn.Linear(hid, dim))
            nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)
            return m

        self.total_enc = nn.ModuleDict({c: _enc() for c in self.total_channels})
        if self.use_partial:
            self.partial_enc = _enc()
            # A learned per-pair id bias so shared partial encoder can tell
            # A-B from C-D; zero-init keeps warm start transparent.
            self.pair_bias = nn.Embedding(self.n_pairs, hid)
            nn.init.zeros_(self.pair_bias.weight)

    def forward(self, data, batch, ng):
        """Return `[N, dim]` scalar-channel bias (or None if nothing active).

        Reads fixed-shape fields off `data`:
          rdf_totals       [ng, n_channels * n_bins]
          rdf_present      [ng, n_channels]
          rdf_partial      [ng, n_pairs * n_bins]
          rdf_pair_present [ng, n_pairs]
        """
        dev = self.total_enc[self.total_channels[0]][0].weight.device \
            if self.total_channels else \
            (self.partial_enc[0].weight.device if self.use_partial else batch.device)
        dtype = next(self.parameters()).dtype
        e = torch.zeros(ng, self.dim, device=dev, dtype=dtype)

        if self.total_channels:
            tot = getattr(data, 'rdf_totals', None)
            pres = getattr(data, 'rdf_present', None)
            if tot is not None:
                tot = tot.view(ng, len(self.total_channels), self.n_bins).to(dtype)
                if pres is not None:
                    pres = pres.view(ng, len(self.total_channels)).to(dtype)
                for ci, cname in enumerate(self.total_channels):
                    out = self.total_enc[cname](tot[:, ci, :])
                    if pres is not None:
                        out = out * pres[:, ci:ci + 1]
                    e = e + out

        if self.use_partial:
            par = getattr(data, 'rdf_partial', None)
            ppres = getattr(data, 'rdf_pair_present', None)
            if par is not None:
                par = par.view(ng, self.n_pairs, self.n_bins).to(dtype)
                pair_ids = torch.arange(self.n_pairs, device=dev)
                # shared encoder: Linear on bins + per-pair id bias, then SiLU
                # + final Linear.  Fold the id bias into the first layer's
                # pre-activation.
                h = self.partial_enc[0](par)                       # [ng, n_pairs, hid]
                h = h + self.pair_bias(pair_ids).unsqueeze(0)      # [1, n_pairs, hid]
                h = self.partial_enc[1](h)                         # SiLU
                out = self.partial_enc[2](h)                       # [ng, n_pairs, dim]
                if ppres is not None:
                    out = out * ppres.view(ng, self.n_pairs, 1).to(dtype)
                e = e + out.sum(dim=1)

        return e[batch]


# Default Gaussian-basis ranges per conditioning property.  Each property
# is expanded over `n_basis` Gaussians evenly spaced on [vmin, vmax]
# (or log-spaced when `log_scale` is True).  Values outside [vmin, vmax]
# still evaluate the Gaussians (no clipping), but only the basis tails
# fire — the model has never seen activations in that regime, so the
# output should be treated as silent extrapolation.  The CLI emits a
# WARNING when any property is supplied outside [vmin, vmax]; see
# `dit/cli.py::_validate_conditioning_ranges`.
#
# H8: `density` widened from [1.5, 5.0] to [1.0, 10.0] to cover the
# realistic range of amorphous oxides (low-density germanate glasses
# can hit ~3 g/cm³; heavy oxides like Ta₂O₅ exceed 8 g/cm³).  Older
# checkpoints persist their own basis-center buffers, so this change
# only affects newly-built models — existing checkpoints keep their
# trained-with ranges.
SCALAR_COND_DEFAULTS = {
    # H13: energy n_basis 32 → 64 (width 0.48 → 0.24 eV/atom).  Conditioning
    # targets in the AL calibration differ by 0.05–0.65 eV/atom; with a
    # 0.48-wide basis a 0.3 eV target shift moves each Gaussian activation
    # by only 0.6 σ, starving the energy axis of gradient relative to the
    # 19×-sharper delta_e_min basis.  Buffers persist per checkpoint, so
    # existing/warm-started models keep their trained 32-basis heads (the
    # shape mismatch means warm starts simply don't copy the old energy MLP);
    # from-scratch builds get the sharper basis.
    'energy':         {'vmin': -15,  'vmax': 0,    'n_basis': 64},
    'density':        {'vmin': 1.0,  'vmax': 10.0, 'n_basis': 32},
    # H13: temperature vmax 5000 → 7800 K so T_eff = ΔE/(1.5 k_B) stays
    # inside the basis for ΔE up to 1.0 eV/atom (the physical-sanity filter
    # ceiling is E_ref + 0.8; 0.8 eV/atom is already T_eff ≈ 6190 K, past
    # the old 5000 K edge).
    'temperature':    {'vmin': 0,    'vmax': 7800,  'n_basis': 32},
    # ΔE above the inherent structure (eV/atom, ≥0): the "distance to the
    # nearest local minimum" that separates local-minimum from thermally-
    # activated configurations (Conditioning v2).  H13: vmax 0.8 → 1.0 —
    # the 2026-07 hot-MD data reaches +0.79 eV/atom, exactly at the old
    # edge Gaussian where the basis half-fires and the response saturates.
    'delta_e_min':    {'vmin': 0,    'vmax': 1.0,   'n_basis': 32},
    'pressure':       {'vmin': 0,    'vmax': 100,   'n_basis': 32},
    'bulk_modulus':   {'vmin': 0,    'vmax': 300,   'n_basis': 32},
    'shear_modulus':  {'vmin': 0,    'vmax': 200,   'n_basis': 32},
    'youngs_modulus': {'vmin': 0,    'vmax': 500,   'n_basis': 32},
    'poisson_ratio':  {'vmin': 0,    'vmax': 0.5,   'n_basis': 32},
    'cooling_rate':   {'vmin': 0,    'vmax': 100,   'n_basis': 32, 'log_scale': True},
}

# ============================================================
# Conv Layers (UVU, Attention, Dual-Cutoff)
# ============================================================
class BesselBasisRB(nn.Module):
    """Radial Bessel basis with polynomial envelope."""
    def __init__(self, cutoff, n_basis=16):
        super().__init__()
        self.cutoff = cutoff
        self.register_buffer('freqs',
            torch.arange(1, n_basis + 1).float() * math.pi / cutoff)
        # Pre-compute the constant prefactor √(2/cutoff) once.
        self._sqrt2_over_cut = math.sqrt(2.0 / cutoff)
        # `freqs` already encodes nπ/cutoff, so forward must not divide by
        # cutoff again (the legacy code did, yielding sin(nπd/c²) — a basis
        # 1/c as resolving as intended; cf. the correct _bessel_basis in
        # graphite.py).  Old whole-module pickles lack this attribute after
        # unpickling (__init__ is not re-run), so they keep the exact
        # legacy math their downstream weights were trained against.
        self._freq_fixed = True

    def forward(self, d):
        d = d.clamp(min=1e-8)
        # Self-heal for old checkpoints saved before _sqrt2_over_cut existed.
        # nn.Module.__setstate__ restores params/buffers but not plain Python
        # attributes set in __init__, so loading a pre-C5 checkpoint leaves
        # this attribute missing.  Recompute on the fly the first time.
        c = getattr(self, '_sqrt2_over_cut', None)
        if c is None:
            c = math.sqrt(2.0 / self.cutoff)
            self._sqrt2_over_cut = c
        arg = self.freqs * d.unsqueeze(-1)
        if not getattr(self, '_freq_fixed', False):
            arg = arg / self.cutoff   # legacy pre-fix checkpoints
        b = arg.sin() * c / d.unsqueeze(-1)
        x = d / self.cutoff
        env = (1 - 6*x**5 + 15*x**4 - 10*x**3).clamp(min=0.0)
        return b * env.unsqueeze(-1)

class EqGate(nn.Module):
    """Equivariant gate activation splitting scalars / higher-L."""
    def __init__(self, irreps):
        super().__init__()
        isc = o3.Irreps([(m, ir) for m, ir in irreps if ir.l == 0])
        igt = o3.Irreps([(m, ir) for m, ir in irreps if ir.l > 0])
        igg = o3.Irreps([(m, (0, 1)) for m, ir in irreps if ir.l > 0])
        self.gate = Gate(isc, [nn.functional.silu] * len(isc),
                         igg, [torch.sigmoid] * len(igg), igt)
        self.irreps_in = self.gate.irreps_in

    def forward(self, x):
        return self.gate(x)

def _make_bn(irreps, factor):
    """Bottleneck irreps: reduce multiplicity by factor."""
    irreps = o3.Irreps(irreps)
    return o3.Irreps([(max(1, m // factor), ir) for m, ir in irreps])

def _make_rmlp(n_radial_basis, radial_neurons, weight_numel):
    """Build radial MLP: n_radial_basis → radial_neurons → weight_numel."""
    dims = [n_radial_basis] + list(radial_neurons) + [weight_numel]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)

def _i0_mul0e(irh):
    """Find offset and total multiplicity of l=0 scalars in `irh`.

    Returns ``(i0, mul0e)`` such that the scalar channel of a flat-layout
    feature tensor lives at ``feature[:, i0 : i0 + mul0e]``.

    M1: asserts that every l=0 block in `irh` is contiguous, i.e. all
    scalar blocks appear before any l>0 block.  Without this guard, a
    config like ``"32x0e + 32x1e + 32x0e"`` would silently fail —
    `_inject_scalar` writes ``feature[:, i0 : i0 + 64]`` and 32 of those
    "scalars" would land in 1e channels, **breaking SO(3) equivariance**.
    The default irreps_hidden (``64x0e + 32x1e``) puts all scalars at
    the head and is unaffected.  Any future config must keep that
    invariant — this assert makes the failure mode loud.
    """
    mul0e = sum(m for m, ir in irh if ir.l == 0)
    idx = 0; i0 = None
    saw_higher_l = False
    contiguous = True
    for m, ir in irh:
        d = m * (2 * ir.l + 1)
        if ir.l == 0:
            if i0 is None:
                i0 = idx
            if saw_higher_l:
                # A scalar block appears AFTER an l>0 block — _inject_scalar's
                # single-slice write would clobber the l>0 channels in
                # between.  Refuse rather than silently break equivariance.
                contiguous = False
        else:
            saw_higher_l = True
        idx += d
    if not contiguous:
        raise ValueError(
            f"_i0_mul0e: irreps_hidden has non-contiguous l=0 channels "
            f"(irreps={irh!r}).  `_inject_scalar` assumes ALL scalar "
            f"multiplets appear before any l>0 multiplet so it can write "
            f"the conditioning into a single contiguous slice.  Reorder "
            f"irreps_hidden to put every l=0 block at the head, e.g. "
            f"'64x0e + 32x1e + 16x2e', or rewrite `_inject_scalar` to "
            f"enumerate per-block slice descriptors.")
    return i0, mul0e

def _inject_scalar(o, c, i0, mul0e):
    """Add scalar conditioning `c` to `o`'s l=0 channel and return the new tensor.

    Returns a new tensor (out-of-place add via F.pad).  Previous versions
    used an in-place slice assignment ``o[:, i0:i0+mul0e] = ...``, which
    works in modern PyTorch but tangles autograd view-tracking when `o`
    has gradient history.  Callers should use the returned value:

        o = _inject_scalar(o, c, i0, mul0e)

    When `c` or `i0` is None this is a no-op and `o` is returned unchanged.
    """
    if c is None or i0 is None:
        return o
    pad_left = i0
    pad_right = o.shape[-1] - i0 - mul0e
    # F.pad pads the last dim by default; left/right are zero-extensions
    # of `c` so its values land exactly on slice [i0:i0+mul0e].
    return o + F.pad(c, (pad_left, pad_right))
