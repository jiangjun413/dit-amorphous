import math
import warnings
import torch
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.nn import Gate
from torch_scatter import scatter
from dit2.model.embeddings import (GaussianBasis, ScalarCondEmbed, BesselBasisRB, EqGate,
                                   MethodEmbed, RDFEmbed,
                                   _make_bn, _make_rmlp, _i0_mul0e, _inject_scalar)

# ============================================================
# Equivariant multi-head attention helpers
# ============================================================
def _build_irh_blocks(irh):
    """Pre-compute per-irrep slice descriptors for `_equivariant_multihead_mix`.

    For each (mul, ir) entry in `irh`, returns a tuple
        (start, end, mul, 2*l+1)
    describing the contiguous block of e3nn's flat layout that holds that
    irrep.  The returned list is consumed by `_equivariant_multihead_mix`
    inside conv `forward` methods.

    The list is intentionally a plain Python list of int tuples — small,
    serialisable, and cheap to iterate (3 irreps * a few ops per layer).
    """
    blocks = []
    offset = 0
    for mul, ir in irh:
        ldim = 2 * ir.l + 1
        blocks.append((offset, offset + mul * ldim, int(mul), int(ldim)))
        offset += mul * ldim
    return blocks


def _heads_align_with_irh(irh, nh: int) -> bool:
    """True iff every irrep multiplicity in `irh` is divisible by `nh`.

    This is the precondition for the per-irrep multi-head split to work
    without leaving a ragged remainder of mul-copies.  When False, the
    conv falls back to a single-scalar (equivariant) weighting.
    """
    return all(int(mul) % int(nh) == 0 for mul, _ in irh)


# Set of conv class names already warned about — keeps the legacy warning
# at one emission per class per process, avoiding log spam in long runs.
_LEGACY_MULTIHEAD_WARNED = set()


def _warn_legacy_multihead_once(cls_name: str) -> None:
    """Emit a one-time warning when an old checkpoint loads a conv that
    lacks the per-irrep multi-head metadata.

    Old checkpoints were trained with a flat multi-head split that broke
    rotation equivariance (different scalar weights on (x, y, z) of the
    same vector irrep).  We refuse to silently keep doing that on load;
    instead the forward path falls through to an equivariant single-
    scalar mean.  Behaviour will differ from training — the user should
    fine-tune or retrain.
    """
    if cls_name in _LEGACY_MULTIHEAD_WARNED:
        return
    _LEGACY_MULTIHEAD_WARNED.add(cls_name)
    import warnings as _w
    _w.warn(
        f"{cls_name}: loaded checkpoint predates the equivariant "
        f"multi-head fix and lacks `_irh_blocks` / `_heads_aligned`. "
        f"Falling back to single-scalar mean-over-heads — equivariant, "
        f"but loses the multi-head expressivity the checkpoint was "
        f"trained with.  Behaviour will differ from training.  Fine-"
        f"tune or retrain to recover proper multi-head attention.",
        RuntimeWarning, stacklevel=3)


def _equivariant_multihead_mix(m, aw, nh: int, irh_blocks):
    """Apply per-head scalar attention weights to an equivariant message
    tensor while preserving SO(3) equivariance.

    For each irrep block in ``irh_blocks`` we split the *multiplicity*
    axis across heads (NOT the flat feature axis).  All (2l+1) components
    of a given multiplicity copy are multiplied by the same head weight,
    which is the requirement for the multiplication to commute with
    rotations.

    The previous implementation used ``m.view(-1, nh, hd)`` over the flat
    feature axis, which sliced individual l>0 multiplets across heads
    (e.g. ``v5_x`` in head 1, ``v5_y, v5_z`` in head 2 for the default
    ``64x0e + 32x1e`` / nh=4 configuration).  Different scalars on the
    three components of the same vector irrep break rotation
    equivariance.

    Args
    ----
    m  : (E, tdim)
        Flat message tensor with irreps described by ``irh_blocks``.
    aw : (E, nh)
        Per-edge per-head attention weights.
    nh : int
        Number of attention heads.
    irh_blocks : list of (start, end, mul, 2l+1)
        Pre-computed by ``_build_irh_blocks``.

    Returns
    -------
    (E, tdim) tensor — same flat layout as ``m``, with each multiplicity
    copy of each irrep scaled by its assigned head weight.
    """
    parts = []
    # aw broadcast helpers — pre-shape once.
    aw_h = aw.unsqueeze(-1)               # (E, nh, 1)         — for scalars
    aw_hl = aw.unsqueeze(-1).unsqueeze(-1) # (E, nh, 1, 1)      — for l>0
    for start, end, mul, ldim in irh_blocks:
        block = m[:, start:end]                       # (E, mul*ldim)
        sub_mul = mul // nh                           # mul-copies per head
        if ldim == 1:
            # l=0 (scalar): split mul across heads, weight per-scalar.
            block = block.view(-1, nh, sub_mul) * aw_h
            parts.append(block.reshape(-1, mul))
        else:
            # l>0: same head weight applies to all (2l+1) components of
            # each mul-copy → preserves rotation equivariance.
            block = block.view(-1, nh, sub_mul, ldim) * aw_hl
            parts.append(block.reshape(-1, mul * ldim))
    return torch.cat(parts, dim=-1)


class ScalarAttn(nn.Module):
    """Scalar attention weights from radial basis + source/dest scalars."""
    def __init__(self, n_radial_basis, sdim, hid=64, nh=4):
        super().__init__()
        self.nh = nh
        self.mlp = nn.Sequential(
            nn.Linear(n_radial_basis + 2 * sdim, hid), nn.SiLU(),
            nn.Linear(hid, nh))

    def forward(self, rb, hs, hd, ei, N):
        logits = self.mlp(torch.cat([rb, hs, hd], -1))
        dst = ei[1]
        mx = scatter(logits, dst, dim=0, reduce='max', dim_size=N)[dst]
        exp = (logits - mx).exp()
        nrm = scatter(exp, dst, dim=0, reduce='sum', dim_size=N)[dst]
        return exp / (nrm + 1e-8)

def _lr_envelope(d, sr_cutoff, lr_cutoff):
    """Smooth LR envelope function — zero at d<=sr_cutoff, smoothly ramps to 1
    across a short transition zone, then smoothly fades to 0 at d=lr_cutoff.

    This eliminates the SR/LR discontinuity: a pair crossing sr_cutoff from
    either side sees a continuous contribution, preventing abrupt force jumps
    that cause atoms to aggregate or fly apart at the boundary.

    Shape:
      d <= sr_cutoff:                              env = 0    (SR handles it)
      sr_cutoff < d <= sr_cutoff + transition:     cosine ramp 0→1
      transition < d <= lr_cutoff:                 poly fade 1→0 at lr_cutoff
      d > lr_cutoff:                               env = 0
    """
    # Transition width: 20% of the SR-LR shell, or 1 Å, whichever is smaller
    transition = min(1.0, 0.2 * (lr_cutoff - sr_cutoff))
    # Ramp-in region: sr_cutoff → sr_cutoff + transition
    ramp_end = sr_cutoff + transition
    # Cosine ramp-in: 0.5*(1 - cos(pi*x)) where x = (d - sr)/transition, x in [0,1]
    x_in = ((d - sr_cutoff) / transition).clamp(0.0, 1.0)
    ramp_in = 0.5 * (1.0 - torch.cos(math.pi * x_in))
    # Polynomial fade-out at lr_cutoff (same shape as NequIP SR envelope)
    x_out = (d / lr_cutoff).clamp(0.0, 1.0)
    fade_out = (1 - 6*x_out**5 + 15*x_out**4 - 10*x_out**3).clamp(min=0.0)
    # Combined envelope: zero below sr, ramps in, then fades out past lr
    env = ramp_in * fade_out
    # Explicitly zero outside [sr_cutoff, lr_cutoff] in case of numerical edge
    env = torch.where(d <= sr_cutoff, torch.zeros_like(env), env)
    env = torch.where(d >= lr_cutoff, torch.zeros_like(env), env)
    return env


class LRAttn(nn.Module):
    """Long-range scalar attention (radial basis + scalar features).

    Multi-head design (B3 fix)
    ──────────────────────────
    Each of the `nh` heads gets its own attention weight AND its own slice
    of the value vector.  Previous revision averaged head weights into a
    single scalar, defeating multi-head.  Now: head h owns
    output channels [h * (odim/nh) : (h+1) * (odim/nh)] which are
    weighted by attention head h.

    Backward compatibility
    ──────────────────────
    If `odim % nh != 0` (older checkpoints built without the divisibility
    constraint), the forward falls back to the legacy mean-over-heads
    behaviour so that existing weights still produce the previous output.
    Newly trained models satisfy `odim % nh == 0` and use the proper
    multi-head path.

    Envelope (C4)
    ─────────────
    Accepts an optional pre-computed envelope tensor `env` (shape (E,)).
    If provided, `lr_rb` is multiplied by `env.unsqueeze(-1)` before use.
    The model's forward computes `env` once and shares it across all
    DualConv layers, avoiding repeated envelope evaluation per layer.

    API simplification (B2)
    ───────────────────────
    The deprecated `precomputed_rb` flag is gone — `lr_rb` is now always
    a (E, n_radial_basis_lr) tensor (the model passes the BesselBasisRB
    output directly).
    """
    def __init__(self, n_radial_basis_lr, sdim, odim, lr_cut=10.0,
                 hid=64, nh=4, sr_cut=None):
        super().__init__()
        self.nh = nh
        self.odim = odim
        self.lr_cut = lr_cut
        self.sr_cut = sr_cut  # set from NequIP_MultiConv for envelope
        # `rb` is kept for backward-compat with old checkpoints that may
        # call self.rb(dist) — the modern path passes lr_rb pre-computed.
        self.rb = BesselBasisRB(lr_cut, n_radial_basis_lr)
        self.a_mlp = nn.Sequential(
            nn.Linear(n_radial_basis_lr + 2 * sdim, hid), nn.SiLU(),
            nn.Linear(hid, nh))
        self.v_mlp = nn.Sequential(
            nn.Linear(n_radial_basis_lr + sdim, hid), nn.SiLU(),
            nn.Linear(hid, odim))
        if odim % nh != 0:
            import warnings as _w
            _w.warn(
                f"LRAttn: odim ({odim}) is not divisible by nh ({nh}); "
                f"falling back to legacy mean-over-heads behaviour for "
                f"this layer.  For proper multi-head attention, choose "
                f"odim such that odim % nh == 0.", RuntimeWarning,
                stacklevel=2)

    def forward(self, hs, ei, lr_rb, N, lr_dist=None, env=None):
        """
        hs       : (N_total, sdim) source/destination scalars
        ei       : (2, E) edge index (src, dst)
        lr_rb    : (E, n_radial_basis_lr) Bessel features for each LR edge
        N        : total node count (graph atoms in the batch)
        lr_dist  : (E,) raw distances — only used when `env` not provided
        env      : (E,) pre-computed envelope, optional
        """
        s, d_ix = ei
        E = lr_rb.size(0)
        nh = self.nh

        # Apply envelope: prefer pre-computed (C4), else compute on demand.
        if env is None and getattr(self, 'sr_cut', None) is not None \
                and lr_dist is not None:
            env = _lr_envelope(lr_dist, self.sr_cut, self.lr_cut)
        if env is not None:
            lr_rb = lr_rb * env.unsqueeze(-1)

        # Per-head attention logits (E, nh)
        logits = self.a_mlp(torch.cat([lr_rb, hs[s], hs[d_ix]], -1))
        # Per-destination softmax (per head independently)
        mx  = scatter(logits, d_ix, dim=0, reduce='max', dim_size=N)[d_ix]
        exp = (logits - mx).exp()
        nrm = scatter(exp, d_ix, dim=0, reduce='sum', dim_size=N)[d_ix] + 1e-8
        aw  = exp / nrm  # (E, nh)

        # Values
        v = self.v_mlp(torch.cat([lr_rb, hs[s]], -1))  # (E, odim)
        odim = self.v_mlp[-1].out_features
        if E > 0:
            if odim % nh == 0:
                head_dim = odim // nh
                v = (v.view(E, nh, head_dim) * aw.unsqueeze(-1)).reshape(E, odim)
            else:
                # Legacy fallback: scalar mean of head weights.
                v = v * aw.mean(-1, keepdim=True)
        # Empty-edge case: v already shape (0, odim); fall through.
        return scatter(v, d_ix, dim=0, reduce='sum', dim_size=N)

# ── Standard NequIP conv (e3nn-based, no bottleneck) ──────
class StandardConvE3(nn.Module):
    """Standard e3nn NequIP conv.

    Already uses sum / √(num_neighbors_avg) scatter normalisation, so the
    √-degree density dependence is implicit (sum scales with N, divided
    by a fixed avg).  We add only the SkipInit stability fix:

      [Fix 2] SkipInit scalar `alpha` (init 0) on the conv contribution
              so at initialisation the layer passes only the residual
              stream (si(h) + zl(z)) and learns to ramp the conv
              contribution on smoothly.
    """
    def __init__(self, iri, irh, irz, ire, nrb, rn, nn_):
        super().__init__()
        self.nn_ = nn_
        self.i0, self.mul0e = _i0_mul0e(irh)
        self.tp = FullyConnectedTensorProduct(iri, ire, irh,
                                               shared_weights=False)
        self.rmlp = _make_rmlp(nrb, rn, self.tp.weight_numel)
        self.si = o3.Linear(iri, irh)
        self.zl = o3.Linear(irz, irh)
        self.gate = EqGate(irh)
        self.pg = o3.Linear(irh, self.gate.irreps_in)
        # [Fix 2] SkipInit scalar — init 0 → conv contribution ramps on
        self.alpha_sr = nn.Parameter(torch.zeros(1))

    def forward(self, h, z, ei, ea, rb, c0=None):
        s, d = ei
        m = self.tp(h[s], ea, self.rmlp(rb))
        scat = scatter(m, d, dim=0, dim_size=h.size(0), reduce='sum') \
               * (1.0 / self.nn_ ** 0.5)
        # [Fix 2] gate the conv contribution with alpha_sr (init 0)
        scat = scat * self.alpha_sr
        o = scat + self.si(h) + self.zl(z)
        o = _inject_scalar(o, c0, self.i0, self.mul0e)
        return self.gate(self.pg(o))

# ── UVU bottleneck conv ───────────────────────────────────
class UVUConv(nn.Module):
    """UVU-bottlenecked NequIP conv.

    Already uses sum / √(num_neighbors_avg) scatter normalisation, so we
    add only the SkipInit stability fix (same rationale as StandardConvE3).
    """
    def __init__(self, iri, irh, irz, ire, nrb, rn, nn_, bf=4):
        super().__init__()
        self.nn_ = nn_
        self.i0, self.mul0e = _i0_mul0e(irh)
        self.ir_bn_in = _make_bn(iri, bf)
        self.ir_bn_out = _make_bn(irh, bf)
        self.V = o3.Linear(iri, self.ir_bn_in)
        self.tp = FullyConnectedTensorProduct(
            self.ir_bn_in, ire, self.ir_bn_out, shared_weights=False)
        self.rmlp = _make_rmlp(nrb, rn, self.tp.weight_numel)
        self.U = o3.Linear(self.ir_bn_out, irh)
        self.si = o3.Linear(iri, irh)
        self.zl = o3.Linear(irz, irh)
        self.gate = EqGate(irh)
        self.pg = o3.Linear(irh, self.gate.irreps_in)
        # [Fix 2] SkipInit scalar — init 0 → conv contribution ramps on
        self.alpha_sr = nn.Parameter(torch.zeros(1))

    def forward(self, h, z, ei, ea, rb, c0=None):
        s, d = ei
        m = self.U(self.tp(self.V(h[s]), ea, self.rmlp(rb)))
        scat = scatter(m, d, dim=0, dim_size=h.size(0), reduce='sum') \
               * (1.0 / self.nn_ ** 0.5)
        # [Fix 2] gate the conv contribution with alpha_sr (init 0)
        scat = scat * self.alpha_sr
        o = scat + self.si(h) + self.zl(z)
        o = _inject_scalar(o, c0, self.i0, self.mul0e)
        return self.gate(self.pg(o))

# ── Attention conv ────────────────────────────────────────
class AttnConv(nn.Module):
    """Attention-weighted NequIP conv.

    The softmax attention normalises weights to sum to 1 per destination
    node, which — unlike graphite's sum/√N — makes the scattered output
    magnitude independent of neighbour count.  During denoising, as local
    density rises the model's repulsive signal should grow with √density
    so that crowded regions get stronger "push-apart" forces.  Without it,
    atoms can aggregate despite low training loss.

      [Fix 1] √(1 + deg) density scaling on the scatter output.
      [Fix 2] SkipInit scalar `alpha_sr` on the conv contribution.
    """
    def __init__(self, iri, irh, irz, ire, nrb, rn, nn_, nh=4):
        super().__init__()
        self.nn_ = float(nn_)
        self.i0, self.mul0e = _i0_mul0e(irh)
        self.nh = nh
        self.sdim = sum(m for m, ir in iri if ir.l == 0)
        self.tdim = sum(m * (2 * ir.l + 1) for m, ir in irh)
        # Equivariance fix: pre-compute per-irrep block layout so the
        # multi-head mix can split the *multiplicity* axis (not the flat
        # feature axis), keeping all (2l+1) components of each l>0
        # multiplet under a single head weight.  See
        # `_equivariant_multihead_mix` for the rationale.
        self._irh_blocks = _build_irh_blocks(irh)
        self._heads_aligned = _heads_align_with_irh(irh, nh)
        self.tp = FullyConnectedTensorProduct(iri, ire, irh,
                                               shared_weights=False)
        self.rmlp = _make_rmlp(nrb, rn, self.tp.weight_numel)
        self.attn = ScalarAttn(nrb, self.sdim, nh=nh)
        self.si = o3.Linear(iri, irh)
        self.zl = o3.Linear(irz, irh)
        self.gate = EqGate(irh)
        self.pg = o3.Linear(irh, self.gate.irreps_in)
        # [Fix 2] SkipInit — init 0 → conv contribution ramps on gradually
        self.alpha_sr = nn.Parameter(torch.zeros(1))

    def forward(self, h, z, ei, ea, rb, c0=None):
        s, d = ei; N = h.size(0)
        m = self.tp(h[s], ea, self.rmlp(rb))
        aw = self.attn(rb, h[s, :self.sdim], h[d, :self.sdim], ei, N)
        # Equivariance fix: per-irrep multi-head mix — each head's scalar
        # weight is shared across the (2l+1) components of every l>0
        # multiplet it owns.  Falls back to a single equivariant scalar
        # when irrep multiplicities aren't divisible by nh, or when the
        # module was loaded from an old checkpoint that predates the fix
        # (no `_irh_blocks` attr) — see `_equivariant_multihead_mix`.
        irh_blocks = getattr(self, '_irh_blocks', None)
        if irh_blocks is not None and getattr(self, '_heads_aligned', False):
            mw = _equivariant_multihead_mix(m, aw, self.nh, irh_blocks)
        else:
            _warn_legacy_multihead_once(type(self).__name__)
            mw = m * aw.mean(-1, keepdim=True)
        scat = scatter(mw, d, dim=0, dim_size=N, reduce='sum')

        # [Fix 1] √-degree density scaling — restores the density
        # dependence that softmax-attention strips away, matching
        # graphite's sum/√N behaviour.
        if s.numel() > 0:
            ones = torch.ones(s.size(0), 1, dtype=scat.dtype,
                              device=scat.device)
            deg  = scatter(ones, d, dim=0, dim_size=N, reduce='sum')
        else:
            deg  = torch.zeros(N, 1, dtype=scat.dtype, device=scat.device)
        scat = scat * (torch.sqrt(1.0 + deg) / (self.nn_ ** 0.5))

        # [Fix 2] SkipInit gating of conv contribution
        scat = scat * self.alpha_sr

        o = scat + self.si(h) + self.zl(z)
        o = _inject_scalar(o, c0, self.i0, self.mul0e)
        return self.gate(self.pg(o))

# ── UVU + Attention conv ──────────────────────────────────
class UVUAttnConv(nn.Module):
    """UVU-bottlenecked attention conv.  Same fixes as AttnConv."""
    def __init__(self, iri, irh, irz, ire, nrb, rn, nn_, bf=4, nh=4):
        super().__init__()
        self.nn_ = float(nn_)
        self.i0, self.mul0e = _i0_mul0e(irh)
        self.nh = nh
        self.sdim = sum(m for m, ir in iri if ir.l == 0)
        self.tdim = sum(m * (2 * ir.l + 1) for m, ir in irh)
        # Equivariance fix: see AttnConv for rationale.
        self._irh_blocks = _build_irh_blocks(irh)
        self._heads_aligned = _heads_align_with_irh(irh, nh)
        self.ir_bn_in = _make_bn(iri, bf)
        self.ir_bn_out = _make_bn(irh, bf)
        self.V = o3.Linear(iri, self.ir_bn_in)
        self.tp = FullyConnectedTensorProduct(
            self.ir_bn_in, ire, self.ir_bn_out, shared_weights=False)
        self.rmlp = _make_rmlp(nrb, rn, self.tp.weight_numel)
        self.U = o3.Linear(self.ir_bn_out, irh)
        self.attn = ScalarAttn(nrb, self.sdim, nh=nh)
        self.si = o3.Linear(iri, irh)
        self.zl = o3.Linear(irz, irh)
        self.gate = EqGate(irh)
        self.pg = o3.Linear(irh, self.gate.irreps_in)
        # [Fix 2] SkipInit
        self.alpha_sr = nn.Parameter(torch.zeros(1))

    def forward(self, h, z, ei, ea, rb, c0=None):
        s, d = ei; N = h.size(0)
        m = self.U(self.tp(self.V(h[s]), ea, self.rmlp(rb)))
        aw = self.attn(rb, h[s, :self.sdim], h[d, :self.sdim], ei, N)
        # Equivariance fix: per-irrep multi-head mix.
        irh_blocks = getattr(self, '_irh_blocks', None)
        if irh_blocks is not None and getattr(self, '_heads_aligned', False):
            mw = _equivariant_multihead_mix(m, aw, self.nh, irh_blocks)
        else:
            _warn_legacy_multihead_once(type(self).__name__)
            mw = m * aw.mean(-1, keepdim=True)
        scat = scatter(mw, d, dim=0, dim_size=N, reduce='sum')

        # [Fix 1] √-degree density scaling
        if s.numel() > 0:
            ones = torch.ones(s.size(0), 1, dtype=scat.dtype,
                              device=scat.device)
            deg  = scatter(ones, d, dim=0, dim_size=N, reduce='sum')
        else:
            deg  = torch.zeros(N, 1, dtype=scat.dtype, device=scat.device)
        scat = scat * (torch.sqrt(1.0 + deg) / (self.nn_ ** 0.5))

        # [Fix 2] SkipInit
        scat = scat * self.alpha_sr

        o = scat + self.si(h) + self.zl(z)
        o = _inject_scalar(o, c0, self.i0, self.mul0e)
        return self.gate(self.pg(o))

# ── Dual-cutoff conv (UVU + attention + long-range attn) ──
class DualConv(nn.Module):
    """Dual-cutoff convolution with short-range equivariant TP + long-range
    scalar attention.

    sigma_t scaling (B1)
    ────────────────────
    Accepts a per-atom noise level tensor `sigma_t` (shape (N,)) and
    smoothly attenuates the long-range contribution at high σ.  Rationale:
    when atoms are far from equilibrium positions (high σ at the start of
    denoising), distant pairs carry less reliable structural information,
    so the LR branch should ramp in as σ decreases.  We use
        lr_scale = exp(-sigma_t)
    which is 1.0 at σ=0 (full LR, late denoising / refine) and decays to
    ~0.47 at σ=σ_max=0.75 (start of denoising).  The decay shape is
    learnable indirectly through alpha_lr.

    Pre-computed envelope (C4)
    ──────────────────────────
    Receives `lr_env` (the SR↔LR cosine envelope) as an argument; the
    model's main forward computes it ONCE and shares it across all conv
    layers, instead of re-evaluating per layer.
    """
    def __init__(self, iri, irh, irz, ire, nrb, rn, nn_, bf=4, nh=4,
                 nrb_lr=12, lr_cut=10.0, sr_cut=None):
        super().__init__()
        self.i0, self.mul0e = _i0_mul0e(irh)
        self.nh = nh
        self.nn_ = float(nn_)
        self.sdim = sum(m for m, ir in iri if ir.l == 0)
        self.tdim = sum(m * (2 * ir.l + 1) for m, ir in irh)
        # Equivariance fix: see AttnConv for rationale.
        self._irh_blocks = _build_irh_blocks(irh)
        self._heads_aligned = _heads_align_with_irh(irh, nh)
        bn_i = _make_bn(iri, bf); bn_o = _make_bn(irh, bf)
        self.V = o3.Linear(iri, bn_i)
        self.tp = FullyConnectedTensorProduct(
            bn_i, ire, bn_o, shared_weights=False)
        self.rmlp = _make_rmlp(nrb, rn, self.tp.weight_numel)
        self.U = o3.Linear(bn_o, irh)
        self.sr_attn = ScalarAttn(nrb, self.sdim, nh=nh)
        self.lr_attn = LRAttn(nrb_lr, self.sdim, self.mul0e,
                               lr_cut, nh=nh, sr_cut=sr_cut)
        # Self-interaction / residual path — always active
        self.si = o3.Linear(iri, irh)
        self.zl = o3.Linear(irz, irh)
        self.gate = EqGate(irh)
        self.pg = o3.Linear(irh, self.gate.irreps_in)

        # SkipInit scalars.  Initialised to zero so the network
        # starts with only the residual path flowing; conv contributions
        # ramp in as training proceeds.  These are cheap (tdim + mul0e
        # learnable scalars per layer) and provide the same warm-start
        # stability graphite gets from its alpha gate.
        self.alpha_sr = nn.Parameter(torch.zeros(1))
        self.alpha_lr = nn.Parameter(torch.zeros(1))

    def forward(self, h, z, ei, ea, rb, c0=None,
                lr_ei=None, lr_rb=None, lr_dist=None, lr_env=None,
                sigma_t=None):
        s, d = ei; N = h.size(0)

        # ── SR equivariant TP message ──────────────────────────────────
        m = self.U(self.tp(self.V(h[s]), ea, self.rmlp(rb)))

        # ── SR attention (softmax-normalised per destination) ──────────
        aw = self.sr_attn(rb, h[s, :self.sdim], h[d, :self.sdim], ei, N)
        # Equivariance fix: per-irrep multi-head mix.  The flat-axis
        # split used previously sliced l>0 multiplets across heads,
        # giving different scalar weights to (x, y, z) of the same
        # vector irrep — this broke SO(3) equivariance.  See
        # `_equivariant_multihead_mix` for the corrected layout.
        irh_blocks = getattr(self, '_irh_blocks', None)
        if irh_blocks is not None and getattr(self, '_heads_aligned', False):
            mw = _equivariant_multihead_mix(m, aw, self.nh, irh_blocks)
        else:
            _warn_legacy_multihead_once(type(self).__name__)
            mw = m * aw.mean(-1, keepdim=True)

        # ── Scatter to destination nodes ──────────────────────────────
        scat = scatter(mw, d, dim=0, dim_size=N, reduce='sum')

        if s.numel() > 0:
            ones = torch.ones(s.size(0), 1, dtype=scat.dtype, device=scat.device)
            deg  = scatter(ones, d, dim=0, dim_size=N, reduce='sum')  # (N,1)
        else:
            deg  = torch.zeros(N, 1, dtype=scat.dtype, device=scat.device)

        # ── √-degree density scaling ───────────────────────────
        density_scale = torch.sqrt(1.0 + deg) / (self.nn_ ** 0.5)
        scat = scat * density_scale

        # SkipInit: gate the conv contribution with alpha_sr ─
        # At init, alpha_sr = 0 → scat contributes nothing; only the
        # residual stream (si(h) + zl(z)) flows through.  Trains to ramp on.
        scat = scat * self.alpha_sr

        # Residual pre-activation
        o = scat + self.si(h) + self.zl(z)

        # ── LR branch ──────────────────────────────────────────────────
        # Always executed for DDP compatibility (empty-edge-safe inside LRAttn).
        if lr_ei is not None and lr_rb is not None:
            lr_c = self.lr_attn(
                h[:, :self.sdim], lr_ei, lr_rb, N,
                lr_dist=lr_dist, env=lr_env)  # (N, mul0e)

            # B1: scale LR contribution by exp(-sigma_t).  At σ=0 this is 1.0
            # (full LR); at high σ the scale shrinks, letting SR dominate.
            # Compute exp(-sigma) in fp32 (sigma_t arrives as fp32 from
            # transforms.py / data._sigma_t) and let autocast promote the
            # subsequent multiply.  Earlier code cast sigma_t to lr_c.dtype
            # under autocast (bf16), claiming it was a no-op — it was not,
            # and the low-σ regime where exp(-σ) ≈ 1 − σ lost precision.
            if sigma_t is not None:
                gate = torch.exp(-sigma_t).unsqueeze(-1)
                lr_c = lr_c * gate.to(lr_c.dtype)

            o = _inject_scalar(o, lr_c * self.alpha_lr, self.i0, self.mul0e)

        o = _inject_scalar(o, c0, self.i0, self.mul0e)

        return self.gate(self.pg(o))

# ── NequIP with swappable conv backbone ───────────────────
class NequIP_MultiConv(nn.Module):
    """NequIP-style denoiser with configurable conv layers and
    multi-property conditioning (energy, density, cooling_rate, etc.).

    Forward signature: model(data) -> displacement
    Conditions are read from data attributes matching scalar_conditions keys.

    conv_type: 'standard_e3', 'uvu', 'attention',
               'uvu_attention', 'dual_cutoff'
    """
    # Periodic-table vocab size: indices 0..118.  Z=0 is reserved as
    # padding_idx (kept fixed at zero), Z=1..118 cover H..Og.
    PERIODIC_TABLE_SIZE = 119

    def __init__(self, cutoff, irreps_node_x='8x0e',
                 irreps_node_z='8x0e',
                 irreps_hidden='64x0e + 32x1e',
                 irreps_edge='4x0e + 4x1e + 2x2e',
                 irreps_out='1x1e',
                 num_convs=3, n_radial_basis=16,
                 radial_neurons=None, num_neighbors=12,
                 conv_type='uvu', uvu_bottleneck_factor=4,
                 n_attention_heads=4,
                 long_range_cutoff=10.0, n_radial_basis_lr=12,
                 scalar_conditions=None,
                 element_latent_dim=None,
                 # Conditioning v2: categorical method tag + RDF encoder.
                 # n_methods=0 / rdf_config=None → legacy behaviour.
                 # sc_zero_init zero-inits the scalar-cond MLPs (warm start).
                 n_methods=0, rdf_config=None, sc_zero_init=False,
                 # Accepted but ignored — kept for backward-compat with
                 # old callers that still pass it.  The embedding is
                 # sized to the full periodic table (PERIODIC_TABLE_SIZE),
                 # so the number of species in any specific dataset is
                 # irrelevant at construction time.
                 num_species=None):
        super().__init__()
        if radial_neurons is None:
            radial_neurons = [16, 64]
        self.cutoff = cutoff
        self.conv_type = conv_type
        irh = o3.Irreps(irreps_hidden)
        ire = o3.Irreps(irreps_edge)
        irx = o3.Irreps(irreps_node_x)
        irz = o3.Irreps(irreps_node_z)
        self.irreps_hidden = irh
        self.irreps_edge = ire
        l_max = max(ir.l for _, ir in ire)

        # ── Embeddings ──
        # Periodic-table-wide tables: data.x carries Z (atomic number) directly.
        # element_latent_dim defaults to irx.dim / irz.dim for legacy callers
        # that built the model from explicit irreps_node_x/z strings.
        edim_x = int(element_latent_dim) if element_latent_dim is not None else irx.dim
        edim_z = int(element_latent_dim) if element_latent_dim is not None else irz.dim
        if edim_x != irx.dim or edim_z != irz.dim:
            raise ValueError(
                f"element_latent_dim={element_latent_dim} must match "
                f"irreps_node_x.dim={irx.dim} and irreps_node_z.dim={irz.dim}. "
                f"Configs should set irreps via element_latent_dim; "
                f"see dit/config.py.")
        self.element_latent_dim = edim_x
        self.embed_x = nn.Embedding(self.PERIODIC_TABLE_SIZE, edim_x, padding_idx=0)
        self.embed_z = nn.Embedding(self.PERIODIC_TABLE_SIZE, edim_z, padding_idx=0)
        self.rbf = BesselBasisRB(cutoff, n_radial_basis)
        self.sh_irreps = o3.Irreps.spherical_harmonics(l_max, p=1)

        # ── Edge feature projections ──
        self.eprojs = nn.ModuleList()
        for m, ir in ire:
            nm = 2 * ir.l + 1
            self.eprojs.append(
                nn.Linear(n_radial_basis * nm, m * nm, bias=False))

        # ── Multi-property conditioning (energy, density, etc.) ──
        mul0e = sum(m for m, ir in irh if ir.l == 0)
        self._i0, self._mul0e = _i0_mul0e(irh)
        self.has_sc = scalar_conditions is not None and len(scalar_conditions) > 0
        if self.has_sc:
            self.sc_emb = ScalarCondEmbed(scalar_conditions, mul0e,
                                          zero_init=sc_zero_init)
        self.cond_names = sorted(scalar_conditions.keys()) if scalar_conditions else []

        # ── Conditioning v2: categorical method tag + RDF encoder ──
        self.n_methods = int(n_methods or 0)
        if self.n_methods > 0:
            self.method_embed = MethodEmbed(self.n_methods, mul0e)
        if rdf_config is not None:
            self.rdf_embed = RDFEmbed(
                n_bins=rdf_config['n_bins'], dim=mul0e,
                n_pairs=rdf_config['n_pairs'],
                total_channels=rdf_config['total_channels'],
                use_partial=rdf_config.get('use_partial', True))

        # ── Long-range RBF (dual_cutoff only) ──
        if conv_type == 'dual_cutoff':
            # H7: guard against degenerate cutoff configurations.  The LR
            # envelope `_lr_envelope` divides by (lr - sr) and would silently
            # produce inf/NaN when lr == sr, or an inverted envelope when
            # lr < sr — neither is recoverable downstream.  Refuse at
            # construction so the user sees the error before training spins
            # up data loaders.
            if not (float(long_range_cutoff) > float(cutoff)):
                raise ValueError(
                    f"NequIP_MultiConv(conv_type='dual_cutoff'): "
                    f"long_range_cutoff ({long_range_cutoff}) must be "
                    f"strictly greater than cutoff ({cutoff}). "
                    f"The LR branch operates in the shell (cutoff, "
                    f"long_range_cutoff] and degenerates to a single "
                    f"radius — or worse, an inverted envelope — when "
                    f"the two are equal or inverted.")
            self.lr_rbf = BesselBasisRB(long_range_cutoff,
                                         n_radial_basis_lr)
            # Store for use in generate() to rebuild LR edges each step
            self.long_range_cutoff = long_range_cutoff
            self.sr_cutoff = cutoff

        # ── Conv layers ──
        nn_ = float(num_neighbors)
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            ii = irx if i == 0 else irh
            if conv_type == 'uvu':
                self.convs.append(UVUConv(
                    ii, irh, irz, ire, n_radial_basis, radial_neurons,
                    nn_, uvu_bottleneck_factor))
            elif conv_type == 'attention':
                self.convs.append(AttnConv(
                    ii, irh, irz, ire, n_radial_basis, radial_neurons,
                    nn_, n_attention_heads))
            elif conv_type == 'uvu_attention':
                self.convs.append(UVUAttnConv(
                    ii, irh, irz, ire, n_radial_basis, radial_neurons,
                    nn_, uvu_bottleneck_factor, n_attention_heads))
            elif conv_type == 'dual_cutoff':
                self.convs.append(DualConv(
                    ii, irh, irz, ire, n_radial_basis, radial_neurons,
                    nn_, uvu_bottleneck_factor, n_attention_heads,
                    n_radial_basis_lr, long_range_cutoff,
                    sr_cut=cutoff))
            elif conv_type == 'standard_e3':
                self.convs.append(StandardConvE3(
                    ii, irh, irz, ire, n_radial_basis, radial_neurons,
                    nn_))
            else:
                # Don't silently fall through to StandardConvE3 — a typo'd
                # conv_type ('attn', 'dual', …) would otherwise train the
                # wrong architecture with no error.
                raise ValueError(
                    f"Unknown conv_type {conv_type!r}; expected one of "
                    "'uvu', 'attention', 'uvu_attention', 'dual_cutoff', "
                    "'standard_e3'.")

        # ── Output head → displacement (default 1x1e = 3D vector) ──
        self.out = o3.Linear(irh, o3.Irreps(irreps_out))

    def _compute_edge_feat(self, ev, ed):
        """Compute edge features: Bessel RBF × spherical harmonics.

        Each (m, ir) block of `irreps_edge` gets its own outer product
        (rb_e ⊗ sh_e[l*l : l*l+nm]) followed by a small Linear projection.
        We use `torch.bmm` because its output is freshly allocated and
        contiguous, so the subsequent `reshape(E, -1)` is a zero-copy view.
        An earlier attempt to fuse the per-l outer products into a single
        elementwise multiply produced non-contiguous slices that forced
        implicit `contiguous()` copies on every reshape — net regression.
        """
        rb = self.rbf(ed)
        sh = o3.spherical_harmonics(
            self.sh_irreps, ev, normalize=True,
            normalization='component')
        E = rb.size(0)
        parts = []
        for idx, (m, ir) in enumerate(self.irreps_edge):
            l = ir.l; nm = 2 * l + 1
            sb = sh[:, l*l:l*l + nm]
            parts.append(self.eprojs[idx](
                torch.bmm(rb.unsqueeze(2), sb.unsqueeze(1)).reshape(
                    E, -1)))
        return torch.cat(parts, -1), rb

    def forward(self, data):
        """Forward pass.

        data must have: x, pos, edge_index, edge_attr, batch
        Conditions are read from data attributes matching cond_names
          (e.g. data.energy, data.density, data.cooling_rate).
        Returns: [N, 3] displacement prediction
        """
        sp = data.x
        ei = data.edge_index
        ev = data.edge_attr

        # ── Edge features ──
        ed = ev.norm(dim=-1)
        # H6: filter at fp32 epsilon, not 1e-6.  Two atoms transiently
        # near-coinciding during high-σ rattling can have legitimate
        # near-zero (but non-pathological) edge norms; the old 1e-6
        # threshold silently dropped those, starving the destination
        # node of a real neighbour and biasing gradients.  Bessel basis
        # already clamps `d` at min=1e-8 internally, so 1e-8 here is
        # safe and only catches numerical noise.
        # Filter degenerate (near-zero-length) edges unconditionally.  The
        # old `if not ok.all():` guard forced a GPU→CPU sync on EVERY forward
        # (training step, eval batch, generation step) to read the bool — the
        # exact per-step sync the rest of this codebase is careful to avoid.
        # Boolean-indexing unconditionally is cheaper than the stall and gives
        # identical output (a no-op copy when nothing is filtered).
        ok = ed > 1e-8
        ei = ei[:, ok]; ev = ev[ok]; ed = ed[ok]
        ea, rb = self._compute_edge_feat(ev, ed)

        # ── Node embeddings ──
        nx = self.embed_x(sp)
        nz = self.embed_z(sp)

        # ── Multi-condition embedding → scalar injection ──
        batch = data.batch if hasattr(data, 'batch') else \
            torch.zeros(sp.size(0), dtype=torch.long, device=sp.device)
        # Use data.num_graphs if available (set by PyG DataLoader during
        # batching, or by mk_data() for single structures) — avoids the
        # batch.max().item() GPU→CPU sync that would stall every forward pass.
        if hasattr(data, 'num_graphs') and data.num_graphs is not None:
            ng = int(data.num_graphs)
        elif hasattr(data, 'ptr') and data.ptr is not None:
            ng = len(data.ptr) - 1
        else:
            # Last-resort fallback: forces a sync, but only when neither
            # num_graphs nor ptr is set — should not happen in practice.
            ng = int(batch.max().item()) + 1

        # Match the activation dtype (from the node embedding) so autocast
        # (bf16/fp16) doesn't get silently promoted to fp32 by an fp32
        # zeros() initializer.  NOTE: `sp = data.x` is int64 atomic
        # numbers — must NOT use sp.dtype here (would make c0 int64 and
        # break the float arithmetic when has_sc=False).
        c0 = torch.zeros(sp.size(0), self._mul0e,
                         device=sp.device, dtype=nx.dtype)
        if self.has_sc:
            sc, pres = {}, {}
            for name in self.cond_names:
                val = getattr(data, name, None)
                if val is not None:
                    if val.dim() == 0:
                        sc[name] = val.unsqueeze(0)
                    elif val.numel() == ng:
                        sc[name] = val.view(ng)
                    else:
                        sc[name] = val.view(ng, -1)[:, 0]
                    # Optional per-graph presence mask ("<name>_present"):
                    # 0 → this condition is OFF for that graph.
                    p = getattr(data, f'{name}_present', None)
                    if p is not None:
                        pres[name] = p.view(-1)[:ng]
            if sc:
                c0 = c0 + self.sc_emb(sc, batch, ng=ng,
                                      present=pres if pres else None)

        # ── Conditioning v2: method tag + RDF (added into the l=0 bias) ──
        # Always execute the heads when enabled (default method id 0 when
        # absent) so every parameter participates on every rank — DDP-safe.
        # Zero-init heads make this a no-op until they train; the presence
        # mask turns an axis off exactly (contribution 0) per graph.
        if getattr(self, 'n_methods', 0) > 0:
            m = getattr(data, 'method', None)
            if m is None:
                m = torch.zeros(ng, dtype=torch.long, device=c0.device)
            mout = self.method_embed(m.view(-1).long())
            mp = getattr(data, 'method_present', None)
            if mp is not None:
                mout = mout * mp.view(-1)[:ng].view(ng, 1).to(mout.dtype)
            c0 = c0 + mout[batch]
        if hasattr(self, 'rdf_embed'):
            c0 = c0 + self.rdf_embed(data, batch, ng)

        # ── Conv stack ──
        h = nx
        if self.conv_type == 'dual_cutoff':
            lr_ei = getattr(data, 'lr_edge_index', None)
            # LR edges always carry full displacement vectors now.
            lr_ev = getattr(data, 'lr_edge_attr', None)
            lr_dist = lr_ev.norm(dim=-1) if lr_ev is not None else None
            # DDP compatibility: lr_attn parameters MUST participate in the
            # forward graph on every rank, even if a particular batch has
            # zero LR edges after rattling.  Otherwise DDP sees "unused
            # parameters" on some ranks and raises the hang-prevention error.
            # We always materialize lr_ei and lr_dist (using empty tensors if
            # needed); scatter/softmax naturally produce zero contributions
            # for empty edge sets, but the MLP weights still get touched.
            if lr_ei is None or lr_dist is None:
                lr_ei = torch.zeros(2, 0, dtype=torch.long, device=ei.device)
                lr_dist = torch.zeros(0, dtype=ea.dtype, device=ea.device)

            lr_rb = self.lr_rbf(lr_dist)  # always compute (handles 0-edge case)

            # ── C4: pre-compute SR↔LR envelope ONCE, share across layers ──
            sr_cut_attr = getattr(self, 'sr_cutoff', None)
            lr_cut_attr = getattr(self, 'long_range_cutoff', None)
            if sr_cut_attr is not None and lr_cut_attr is not None:
                lr_env = _lr_envelope(lr_dist, sr_cut_attr, lr_cut_attr)
            else:
                lr_env = None  # let LRAttn fall back to its own sr_cut/lr_cut

            # ── B1: build per-atom sigma_t tensor from data._sigma_t ──
            # data._sigma_t is shape (ng,) (set by RattleParticles or generate())
            # or (1,) for single-graph inference.  We broadcast to per-atom.
            sigma_t_per_atom = None
            sigma_t_raw = getattr(data, '_sigma_t', None)
            if sigma_t_raw is not None:
                # Tolerate scalars / 0-d / 1-d tensors.
                if not torch.is_tensor(sigma_t_raw):
                    sigma_t_raw = torch.tensor([float(sigma_t_raw)],
                                               device=sp.device,
                                               dtype=ea.dtype)
                if sigma_t_raw.dim() == 0:
                    sigma_t_per_atom = sigma_t_raw.expand(sp.size(0))
                elif sigma_t_raw.numel() == 1:
                    sigma_t_per_atom = sigma_t_raw.view(()).expand(sp.size(0))
                elif sigma_t_raw.numel() == ng:
                    sigma_t_per_atom = sigma_t_raw[batch]  # (N_total,)
                elif sigma_t_raw.numel() == sp.size(0):
                    sigma_t_per_atom = sigma_t_raw  # already per-atom
                # else: shape mismatch → leave None (no scaling)

            for conv in self.convs:
                h = conv(h, nz, ei, ea, rb, c0,
                         lr_ei=lr_ei, lr_rb=lr_rb, lr_dist=lr_dist,
                         lr_env=lr_env, sigma_t=sigma_t_per_atom)
        else:
            for conv in self.convs:
                h = conv(h, nz, ei, ea, rb, c0)

        return self.out(h)
