import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
from ase import Atoms
from ase.data import atomic_numbers as ASE_Z
from torch_geometric.data import Data

from dit2.generation.neighbor_list import VL, build_nl
from dit2.generation.structure import create_struct_auto
from dit2.model.graphite import GraphiteModelAdapter
from dit2.compat import _is_adapter


def cuda_reset_if_needed(dv):
    """After a device-side assert, the CUDA context is poisoned.
    Reset the device so subsequent runs don't cascade-fail."""
    if dv.type != "cuda":
        return
    try:
        torch.cuda.synchronize(dv)
    except RuntimeError:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        torch.cuda.reset_peak_memory_stats(dv)
        try:
            _ = torch.zeros(1, device=dv)
            torch.cuda.synchronize(dv)
        except Exception:
            pass  # device truly broken, user must restart process


def _mps_available() -> bool:
    """True iff this build has a working Apple-Silicon MPS backend."""
    try:
        return bool(torch.backends.mps.is_available()
                    and torch.backends.mps.is_built())
    except AttributeError:
        return False


def _is_gpu_device(dv) -> bool:
    """True for any GPU-class device (CUDA or Apple Silicon MPS)."""
    return dv.type in ("cuda", "mps")


def _empty_gpu_cache(dv) -> None:
    """Release cached blocks on the device's allocator (no-op on CPU)."""
    if dv.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    elif dv.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

###############################################################################
# DATA VALIDATION
###############################################################################
def validate_data(data, model):
    """Check data.x indices are within the model's embedding range.
    Raises ValueError with a clear message instead of a CUDA assert.

    Handles BOTH the legacy graphite-style `embed_node_x` and the modern
    `NequIP_MultiConv.embed_x` attribute names.  If neither is present,
    falls back to scanning for any nn.Embedding module.
    """
    n_sp = None
    # Priority 1 / 2: named species embedding (legacy or modern)
    for mod in model.modules():
        emb = getattr(mod, "embed_node_x", None)
        if emb is None:
            emb = getattr(mod, "embed_x", None)
        if isinstance(emb, nn.Embedding):
            n_sp = emb.num_embeddings
            break
    # Priority 3: first nn.Embedding found in the model
    if n_sp is None:
        for mod in model.modules():
            if isinstance(mod, nn.Embedding):
                n_sp = mod.num_embeddings
                break
    if n_sp is None:
        return  # cannot validate — no embedding found
    xmax = int(data.x.max().item()) if data.x.numel() > 0 else -1
    xmin = int(data.x.min().item()) if data.x.numel() > 0 else 0
    if xmax >= n_sp or xmin < 0:
        # `data.x` now holds raw atomic numbers Z (no LabelEncoder); the
        # embedding is sized to the periodic table (Z = 0..118). Surface
        # the offending Zs against the training-Z set so the user can
        # see immediately which symbol is unsupported.
        training_z = getattr(model, 'training_z', None) or getattr(
            getattr(model, 'model', None), 'training_z', None)
        msg = (
            f"data.x has atomic numbers Z in [{xmin}, {xmax}] but the "
            f"model embedding only has {n_sp} rows (valid Z: 0–{n_sp-1}).\n"
            f"Unique Z in input: {sorted(data.x.unique().cpu().tolist())}\n")
        if training_z:
            msg += (
                f"Model was trained on Z = {sorted(int(z) for z in training_z)}. "
                f"Any input Z outside the embedding range (or outside the "
                f"trained set) cannot be generated — adjust the composition "
                f"in the sidebar / config.")
        else:
            msg += (
                "The model has no `training_z` stamp; check that the "
                "checkpoint was produced by the current training code.")
        raise ValueError(msg)

###############################################################################
# DATA + CLEANUP   (FIX: free_mask stored on Data)
###############################################################################
def mk_data(atoms, all_model_elements, energy, dv, fixed_mask=None,
            training_z=None, region=None,
            method=None, temperature=None, delta_e_min=None, rdf=None,
            cond_weights=None):
    """Create PyG Data with optional fixed_mask.

    `data.x` carries raw atomic numbers Z (int64); the model embeds them
    through a periodic-table-wide table.  The two element-list parameters
    are kept for sanity-checking the input composition against what the
    model was trained on.

    Parameters
    ----------
    all_model_elements : list of element symbols
        Fallback element list (used only if training_z is not provided).
    training_z : list of int, optional
        Sorted atomic-number list persisted on the model
        (model.training_z).  When provided, bounds the set of Z that
        the input is allowed to contain.  Fallback to all_model_elements
        only for backward compatibility with models saved before
        training_z was persisted.
    fixed_mask : np.ndarray (N,3) bool, optional
        True = atom coordinate is FIXED (displacement zeroed).
        None = all atoms free.
    """
    # Validate `energy` before anything else — a NaN here silently propagates
    # through the scalar-conditioning MLP and produces NaN displacements that
    # are hard to trace back.  `energy=None` is ALLOWED (Conditioning v2):
    # it generates UNconditioned on energy — a zero placeholder rides on
    # data.energy (models read the tensor unconditionally) with
    # data.energy_present = 0 masking its contribution to exactly zero.
    if energy is not None and not math.isfinite(float(energy)):
        raise ValueError(
            f"mk_data: energy must be a finite real number or None "
            f"(= unconditioned), got {energy!r}.")

    if training_z is not None:
        allowed_z = set(int(z) for z in training_z)
    else:
        allowed_z = set(ASE_Z[e] for e in all_model_elements)

    x_raw = atoms.numbers
    unknown = set(int(z) for z in x_raw) - allowed_z
    if unknown:
        z_to_sym = {v: k for k, v in ASE_Z.items()}
        bad = [f"{z_to_sym.get(z, '?')}(Z={z})" for z in unknown]
        sym_list = [z_to_sym.get(z, f'Z{z}') for z in sorted(allowed_z)]
        raise ValueError(
            f"Atoms contain elements {bad} not in model's training set "
            f"{sym_list}. "
            f"Either fix the input POSCAR/composition or retrain the model "
            f"with the broader element set.")

    x = torch.from_numpy(x_raw.astype(np.int64))
    data = Data(x=x, pos=torch.tensor(atoms.positions, dtype=torch.float32),
                cell=torch.tensor(np.array(atoms.cell), dtype=torch.float32),
                pbc=atoms.pbc, numbers=atoms.numbers,
                energy=torch.tensor([0.0 if energy is None else energy],
                                    dtype=torch.float32),
                batch=torch.zeros(len(x), dtype=torch.long), num_graphs=1).to(dv)
    # Conditioning weights (importance multipliers riding on the presence
    # masks): 1 = trained-default strength, 0 = off, >1 amplifies the axis.
    _cw = dict(cond_weights or {})
    data.energy_present = torch.tensor(
        [0.0 if energy is None else _cw.get('energy', 1.0)],
        dtype=torch.float32).to(dv)

    # Store free_mask (True = FREE, i.e. allowed to move)
    if fixed_mask is not None and np.any(fixed_mask):
        data.free_mask = torch.tensor(~fixed_mask, dtype=torch.float32).to(dv)  # (N, 3)
    else:
        data.free_mask = None

    # Optional per-atom region / phase label (Tier-B native interface
    # conditioning).  Only region-aware models read `data.region`; the
    # bulk model ignores it, so attaching it is always safe.
    if region is not None:
        data.region = torch.as_tensor(np.asarray(region), dtype=torch.long).to(dv)

    # ── Conditioning v2 (all optional; models without the heads ignore them) ──
    # Every axis is independently optional: a None value emits a zero
    # placeholder + presence 0, so v2 models generate genuinely
    # UNconditioned on that axis (contribution exactly zero).
    data.method = torch.tensor([0 if method is None else int(method)],
                               dtype=torch.long).to(dv)
    data.method_present = torch.tensor(
        [0.0 if method is None else _cw.get('method', 1.0)],
        dtype=torch.float32).to(dv)
    data.temperature = torch.tensor(
        [0.0 if temperature is None else float(temperature)],
        dtype=torch.float32).to(dv)
    data.temperature_present = torch.tensor(
        [0.0 if temperature is None else _cw.get('temperature', 1.0)],
        dtype=torch.float32).to(dv)
    data.delta_e_min = torch.tensor(
        [0.0 if delta_e_min is None else float(delta_e_min)],
        dtype=torch.float32).to(dv)
    data.delta_e_min_present = torch.tensor(
        [0.0 if delta_e_min is None else _cw.get('delta_e_min', 1.0)],
        dtype=torch.float32).to(dv)
    # dit2: density condition — intrinsic to the cell being generated, so it
    # is computed here and always presented (models without a density head
    # simply never read it).  Weight via physics.cond_weights['density'].
    _rho = float(atoms.get_masses().sum()
                 / abs(np.linalg.det(np.array(atoms.cell))) * 1.66053906660)
    data.density = torch.tensor([_rho], dtype=torch.float32).to(dv)
    data.density_present = torch.tensor(
        [_cw.get('density', 1.0)], dtype=torch.float32).to(dv)
    if rdf is not None:
        # `rdf` = dict with any of: 'totals' [n_ch, nb], 'present' [n_ch],
        # 'partial' [n_pairs, nb], 'pair_present' [n_pairs].  Fixed-shape
        # tensors + presence masks, matching the training collate layout
        # ([1, K] so the model's view(ng, ...) reshapes line up).  The rdf
        # conditioning weight scales the presence masks.
        _w_rdf = _cw.get('rdf', 1.0)
        for k, name in (('totals', 'rdf_totals'), ('present', 'rdf_present'),
                        ('partial', 'rdf_partial'),
                        ('pair_present', 'rdf_pair_present')):
            if k in rdf and rdf[k] is not None:
                arr = np.asarray(rdf[k], dtype=np.float32).reshape(1, -1)
                if k in ('present', 'pair_present') and _w_rdf != 1.0:
                    arr = arr * _w_rdf
                setattr(data, name, torch.from_numpy(arr).to(dv))

    return data

_KEEP = frozenset({"x","pos","cell","pbc","numbers","energy","batch","num_graphs",
                    "edge_index","edge_attr","dx","free_mask","region",
                    # Preserve LR edges across generate steps for dual_cutoff
                    "lr_edge_index","lr_edge_attr",
                    # Preserve noise level for dual_cutoff sigma_t scaling
                    "_sigma_t",
                    # Conditioning v2 (method tag / T / ΔE / RDF channels)
                    "method","temperature","delta_e_min","density",
                    "energy_present","method_present",
                    "temperature_present","delta_e_min_present","density_present",
                    "rdf_totals","rdf_present","rdf_partial","rdf_pair_present"})
def _clean(data):
    for k in [k for k in data.keys() if k not in _KEEP]: delattr(data, k)

###############################################################################
# GENERATION LOOP  (FIX: poscar_start_sigma + free_mask enforcement)
###############################################################################
def _get_training_z(model):
    """Extract the persisted training_z list (sorted atomic numbers) from a
    model, handling the GraphiteModelAdapter wrapper.  Returns None if the
    model predates training_z persistence — callers should fall back to
    deriving the list from model_elements in that case.
    """
    inner = model.model if _is_adapter(model) else model
    tz = getattr(inner, 'training_z', None)
    return list(tz) if tz is not None else None


# Default σ_max used for training (matches `sigma_max` fallback in
# config_to_dict).  Used as the inference-time fallback when a model
# predates `training_sigma_max` persistence.
_DEFAULT_TRAINING_SIGMA_MAX = 0.75
# Lower bound of σ during training (matches `RattleParticles.__call__`'s
# `uniform_(0.001, sigma_max)`).  Used as the σ floor in the refine phase
# of generate(), so the model is never queried at exactly σ=0 — outside
# the training distribution.
_TRAINING_SIGMA_FLOOR = 0.001


def _get_training_sigma_max(model) -> float:
    """Extract the persisted σ_max the model was trained with.

    Falls back to the default training σ_max (0.75) for older models
    that predate `training_sigma_max` persistence.  Used by `generate()`
    to keep the inference schedule inside the training distribution.
    """
    inner = model.model if _is_adapter(model) else model
    sm = getattr(inner, 'training_sigma_max', None)
    if sm is None:
        return _DEFAULT_TRAINING_SIGMA_MAX
    return float(sm)


def _detect_dual_cutoff(model):
    """Return (sr_cutoff, lr_cutoff) if model has dual_cutoff convs, else (None, None).

    Works for GraphiteModelAdapter-wrapped and raw NequIP_MultiConv models.

    If conv_type=='dual_cutoff' but the explicit attributes are missing, this
    function attempts to recover them from BesselBasisRB modules.  If recovery
    also fails, a LOUD warning is emitted (an undetected dual-cutoff model
    will silently run with empty LR edges, severely degrading quality).
    """
    inner = model.model if _is_adapter(model) else model
    ct = getattr(inner, 'conv_type', None)
    if ct != 'dual_cutoff':
        return None, None
    sr = getattr(inner, 'sr_cutoff', None)
    lr = getattr(inner, 'long_range_cutoff', None)
    # Fallback recovery from Bessel modules — same logic as _compat_patch_dual_cutoff.
    if sr is None:
        sr_rbf = getattr(inner, 'rbf', None)
        if sr_rbf is not None and hasattr(sr_rbf, 'cutoff'):
            sr = float(sr_rbf.cutoff)
        else:
            sr = getattr(inner, 'cutoff', None)
    if lr is None:
        lr_rbf = getattr(inner, 'lr_rbf', None)
        if lr_rbf is not None and hasattr(lr_rbf, 'cutoff'):
            lr = float(lr_rbf.cutoff)
    if sr is None or lr is None:
        import warnings as _w
        _w.warn(
            "Model has conv_type='dual_cutoff' but sr_cutoff/long_range_cutoff "
            "could NOT be detected from the model.  The long-range branch will "
            "receive an EMPTY edge set during generation, silently dropping the "
            "LR contribution the model was trained with.  This will significantly "
            "degrade output quality.  Set `model.sr_cutoff` and "
            "`model.long_range_cutoff` on the saved model, or rebuild it.",
            RuntimeWarning, stacklevel=2)
        return None, None
    return float(sr), float(lr)


def _split_edges_dual(edge_index, edge_attr, sr_cutoff):
    """Split a single edge set (built at lr_cutoff) into SR and LR components.

    SR: d <= sr_cutoff         → keep displacement vectors for equivariant TP
    LR: sr_cutoff < d          → keep displacement vectors (correct for small
                                  cells where cell/2 < lr_cutoff and
                                  multiple periodic images appear as distinct
                                  edges between the same (i, j) pair).
    Returns (sr_ei, sr_ea, lr_ei, lr_ea).
    """
    d = edge_attr.norm(dim=-1)
    sr_mask = d <= sr_cutoff
    lr_mask = ~sr_mask
    sr_ei = edge_index[:, sr_mask]
    sr_ea = edge_attr[sr_mask]
    lr_ei = edge_index[:, lr_mask]
    lr_ea = edge_attr[lr_mask]
    return sr_ei, sr_ea, lr_ei, lr_ea


def _enforce_min_dist(pos, cell, pbc, numbers, min_dist, dv,
                      free_mask=None, n_iters=4):
    """Excluded-volume projection: push apart any atom pair closer than
    `min_dist`, iterated a few times (Jacobi-style overlap removal).

    This is a hard *physical* steric floor applied AFTER the model's
    displacement each step — unlike `max_disp_*` (which only caps the
    per-step step length, preserving direction), it directly forbids the
    catastrophic overlaps a mis-calibrated score can drive (e.g. a
    dual-cutoff model queried at low σ on a random init).  `min_dist`
    should sit below the shortest real bond (e.g. 1.0–1.2 Å for oxides,
    vs Si-O ≈ 1.6 Å) so it only catches unphysical contacts and never
    fights legitimate coordination or the diffusion noise.

    PBC-aware (uses the same neighbor-list builder as generation, so
    periodic images are handled).  Respects `free_mask` — fixed atoms are
    never moved; for a fixed/free pair only the free atom is pushed (its
    half of the overlap; the remainder resolves over the iterations).
    """
    md2 = float(min_dist)
    for _ in range(int(n_iters)):
        ei, ea, _pw = build_nl(pos, cell, pbc, numbers, md2, dv)
        if ei.numel() == 0:
            break
        d = ea.norm(dim=1)
        # ── Audit finding F5 (2026-08-06): count each unordered pair ONCE ──
        # Every neighbour list in this package is DIRECTED-SYMMETRIC: an
        # overlapping pair (a,b) is returned twice, as (a,b,+r) and
        # (b,a,-r).  The two entries used to contribute the SAME ±half to
        # the SAME atoms, so each partner received the FULL overlap and the
        # pair separation grew by 2·overlap: a 0.5 Å contact was thrown out
        # to 1.70 Å instead of the intended 1.10 Å (min_dist = 1.1) — well
        # inside the real Si-O / Ge-O bond window, i.e. silently wrong
        # geometry rather than a crash.  Selecting i < j keeps exactly one
        # representative of each reciprocal pair while preserving every
        # DISTINCT periodic image (image edges come as (a,b,+S) / (b,a,−S),
        # so one survives per image).  Verified symmetric on all four NL
        # backends (ASE/CPU ortho + triclinic, _nl_br, _nl_cl).
        # i == j self-image edges are dropped by i < j; that is a strict
        # no-op, because -half and +half landed on the same row and
        # cancelled exactly even before this fix.
        m = (d < md2) & (d > 1e-6) & (ei[0] < ei[1])
        if not bool(m.any()):
            break
        i, j = ei[0, m], ei[1, m]
        vec = ea[m]
        dist = d[m].unsqueeze(1)
        unit = vec / dist                      # points i → j (min image)
        overlap = (md2 - dist).clamp(min=0.0)  # >0 for offending pairs
        # Half the overlap to each partner (symmetric push-apart).
        half = 0.5 * overlap * unit
        corr = torch.zeros_like(pos)
        corr.index_add_(0, i, -half)           # move i away from j
        corr.index_add_(0, j, half)            # move j away from i
        if free_mask is not None:
            # Don't move pinned atoms; the free partner still gets pushed.
            corr = corr * free_mask
        pos = pos + corr
    return pos


def _build_sigma_schedule(sigma_top: float, sigma_floor: float, n: int,
                          mode: str = "edm", rho: float = 7.0,
                          device=None) -> torch.Tensor:
    """Return a length-`n` descending σ schedule on `device`.

    Modes
    -----
    ``"edm"`` (Karras et al. 2022, ``arXiv:2206.00364``) — power-law schedule
        σ_i = (σ_max^(1/ρ) + i/(n-1) · (σ_min^(1/ρ) − σ_max^(1/ρ)))^ρ
        with ρ=7.  Concentrates steps at low σ where the score 1/σ
        becomes large.  This is the schedule used by EDM, DDIM, and most
        modern image diffusion variants — matches the paper claim.

    ``"linear"`` — ``torch.linspace(sigma_top, sigma_floor, n)``.  The
        pre-C5 default.  Kept for bit-exact reproducibility of old runs.

    Both schedules start at exactly ``sigma_top`` (the model's training
    σ_max) and end at exactly ``sigma_floor`` (the σ floor used at the
    end of refine).
    """
    if n <= 0:
        return torch.empty(0, device=device)
    if n == 1:
        return torch.tensor([sigma_top], device=device)
    if mode == "linear":
        return torch.linspace(sigma_top, sigma_floor, n, device=device)
    if mode != "edm":
        raise ValueError(
            f"sigma_schedule must be 'edm' or 'linear', got {mode!r}.")
    # EDM power-law schedule.
    inv_rho = 1.0 / float(rho)
    t = torch.linspace(0.0, 1.0, n, device=device)
    a = float(sigma_top) ** inv_rho
    b = float(sigma_floor) ** inv_rho
    return (a + t * (b - a)) ** float(rho)


def generate(model, data, atoms, noise, refine, snap_every, plot_every,
             gnn_cut, vskin, device, sigmas=None,
             stop_fn=None, on_snap=None, on_plot=None,
             poscar_start_sigma=None,
             max_disp_denoise=0.0, max_disp_refine=0.0,
             sigma_schedule="edm", sigma_rho=7.0,
             repaint=False, known_pos=None, min_dist_constraint=0.0,
             com_detrend="xy", cfg_weight=1.0,
             refine_sigma=None):
    """
    Parameters
    ----------
    cfg_weight : float
        Classifier-free-guidance weight w.  1.0 (default) = plain
        conditional sampling, one model call per step.  w != 1.0 runs TWO
        forwards per step — conditioned (the presence masks as supplied)
        and unconditioned (every conditioning presence mask zeroed) — and
        extrapolates  disp = disp_u + w * (disp_c - disp_u).  w > 1
        SHARPENS the conditional distribution (tighter adherence to the
        ΔE / method / RDF targets, at the cost of diversity); w < 1
        softens it.  Training with presence dropout (cond_dropout /
        RDFPresenceDropout) is what makes the unconditioned branch
        in-distribution — models trained without dropout should keep
        w = 1.  If no conditioning axis is active, guidance is a no-op.
    com_detrend : str
        COM-detrend mode for the FREE atoms when a free_mask pins part of
        the structure: "xy" (default) subtracts the mean displacement only
        along the in-plane axes — a uniform in-plane translation is a true
        zero mode under PBC, while net motion along the stack axis (film
        settling toward / expanding away from a pinned substrate) is
        physical and must be allowed.  "xyz" detrends all three axes (the
        pre-audit behaviour — appropriate for very thick films, ≫ the
        model's receptive field, whose far region otherwise accumulates a
        spurious constant drift and collapses onto the substrate; observed
        on a 105 Å a-GeO₂ film).  "off" disables the detrend entirely.
        Validated on a 624-atom quartz/a-GeO₂ interface: "xyz" gave 23%
        CN4 (over-compressed film), "xy" 75%, "off" 71%.
        Any subset of "xyz" is accepted (e.g. "yz" for a stack axis of x;
        `interface_main` resolves this automatically from stack_axis).
        The unmasked (bulk) branch always detrends all axes — every axis
        is periodic there, so a uniform translation is always spurious.
    min_dist_constraint : float
        Excluded-volume steric floor (Å) enforced after every step (see
        `_enforce_min_dist`).  0 = off.  A hard *physical* constraint that
        forbids atom pairs closer than this — the reliable way to keep a
        mis-calibrated score (e.g. a dual-cutoff model on a random init at
        low σ) from collapsing the structure.  Set below the shortest real
        bond (≈1.0–1.2 Å for oxides) so it only removes unphysical
        overlaps.  Unlike `max_disp_*` it constrains geometry, not step
        length, so it cannot be defeated by a persistent collapse
        direction.
    repaint : bool
        RePaint-style conditioning of the *fixed* region (Lugmayr et al.
        2022, arXiv:2201.09865), used for interface / inpainting
        generation.  When True and `data.free_mask` pins a subset of
        atoms (a substrate / known phase), those atoms are re-noised to
        the *scheduled* σ at the start of every denoise step:

            pos[fixed] = known_pos[fixed] + σ_i · ε

        The plain free_mask path (repaint=False) instead holds the fixed
        atoms at σ=0 (perfectly ordered) throughout.  At high σ that is
        out-of-distribution for any edge crossing the interface — the
        free endpoint is heavily noised while the fixed endpoint is
        pristine, a configuration the denoiser never saw in training.
        Re-noising the known region to the same σ restores a homogeneous
        noise field across the interface so the score the model predicts
        for the free atoms is in-distribution.  During refine the fixed
        atoms are clamped to the σ-floor (≈ known_pos), so the substrate
        ends pristine.  Displacements the model emits for the fixed atoms
        are still zeroed by `free_mask`, so re-noising governs their
        position, not the model.
    known_pos : torch.Tensor (N, 3) or None
        Target (pristine) positions of the known/fixed atoms used by the
        RePaint re-noising above.  None → captured from `data.pos` at
        entry (correct when the fixed atoms start at their target).
    poscar_start_sigma : float or None
        Controls noise when starting from a POSCAR.
        - None  → full noise schedule (original behaviour, for random init)
        - 0.0   → skip noise phase entirely, only refine
        - 0.01–0.1 → light noise, good for relaxing a POSCAR
        Noise steps where sigma > poscar_start_sigma are SKIPPED.
    max_disp_denoise : float
        Per-atom max displacement (Å) per denoise step. 0 = no limit.
    max_disp_refine : float
        Per-atom max displacement (Å) per refine step. 0 = no limit.
    sigma_schedule : str
        "edm" (default) — Karras-et-al. ρ-power schedule, concentrates
        compute at low σ where the score grows as 1/σ.  Matches the
        paper's claim of an EDM schedule.
        "linear" — torch.linspace from σ_max to σ_min.  Pre-C5 default;
        kept for bit-exact comparison with old runs.
    sigma_rho : float
        ρ exponent for "edm" mode (default 7, the Karras paper value).
        Higher ρ → more compute at low σ.  Ignored for "linear".

    For dual_cutoff models, the model's sr_cutoff and long_range_cutoff
    are detected automatically.  VL is built at lr_cutoff (the outer radius)
    so that BOTH SR and LR edges can be split from the same neighbor list
    at every step — correctly tracking edges that migrate between SR and LR
    as atoms move during denoising.  The `gnn_cut` argument is overridden
    to lr_cutoff in that case.
    """
    # ── REFINE-σ FLOOR (ported from dit6 2026-09-01) ─────────────────────────
    # WHY THIS WAS ADDED.  multiscan11 could not match refine σ across packages:
    # dit5/6/7 ran the campaign-validated 0.020 while dit2 was PINNED at its
    # training floor 0.001, because this function had no knob.  The docstring of
    # build_multiscan11.py records that as "a property of the package, not a
    # setting", and every dit2-vs-dit6 conditioning comparison in that campaign
    # carries it.  It is now a setting in both packages, so the comparison can be
    # run matched -- which is the only way to tell whether dit2's ΔE-conditioning
    # advantage lives in its WEIGHTS or in its sampler terminating in the blind
    # regime (cos(out,dx) ≈ 0.01 at σ=0.001, where the output is almost entirely
    # conditioning drift).
    #
    # DEFAULT IS UNCHANGED BEHAVIOUR.  None / "auto" -> _TRAINING_SIGMA_FLOOR, so
    # every existing dit2 config and every banked dit2 number reproduces exactly.
    if refine_sigma is None or (isinstance(refine_sigma, str)
                                and refine_sigma.strip().lower() == "auto"):
        sig_floor = _TRAINING_SIGMA_FLOOR
    else:
        sig_floor = float(refine_sigma)
    _s_top = _get_training_sigma_max(model)
    if not (0.0 < sig_floor < _s_top):
        raise ValueError(
            f"refine_sigma must satisfy 0 < refine_sigma < σ_max ({_s_top}); "
            f"got {refine_sigma!r} -> {sig_floor}.  It is the σ the trajectory "
            f"TERMINATES at; σ=0 is outside the training distribution "
            f"(RattleParticles samples U({_TRAINING_SIGMA_FLOOR}, σ_max)).")

    total = noise + refine; na = len(atoms)

    # ── Detect dual_cutoff and switch VL to lr_cutoff so we get BOTH SR and LR ──
    sr_cut_dual, lr_cut_dual = _detect_dual_cutoff(model)
    is_dual = sr_cut_dual is not None
    if is_dual:
        # Use lr_cutoff so VL's neighbor list covers the full radius.
        # We will split SR/LR from it at every step.
        # Scale vskin by lr_cut / sr_cut so the relative skin
        # (skin / cutoff) stays the same — otherwise sidebar-tuned
        # `verlet_skin` (which assumed cutoff = sr_cut) shrinks
        # relatively and triggers far more frequent neighbor-list
        # rebuilds than the user expects.
        if sr_cut_dual and sr_cut_dual > 0:
            vskin = float(vskin) * (lr_cut_dual / float(sr_cut_dual))
        gnn_cut = lr_cut_dual

    vl = VL(gnn_cut, vskin, device, na=na)
    if sigmas is None and noise > 0:
        # Build the schedule inside the training distribution: top is the
        # persisted training σ_max (default 0.75), floor is the training
        # σ floor (0.001).  The previous schedule started at 1.0 — above
        # σ_max — so the early denoise steps queried the LR sigma_t
        # scaling at OOD σ values (e.g. exp(-1.0) ≈ 0.37 vs the lowest
        # value the model saw, exp(-0.75) ≈ 0.47).
        # C5: schedule shape selectable via `sigma_schedule` (default
        # "edm" with ρ=7).
        sigma_top = _get_training_sigma_max(model)
        sigmas = _build_sigma_schedule(
            sigma_top, sig_floor, noise,
            mode=sigma_schedule, rho=sigma_rho, device=device)

    # ── Pre-transfer sigmas to CPU once ───────────────────────────────
    # sigmas[si].item() inside the hot loop forces a GPU→CPU sync every
    # step; for 3000-step denoise runs that's 3000 syncs.  Since sigmas
    # is just a schedule of scalars, convert to a plain Python list once.
    sigmas_cpu = sigmas.detach().cpu().tolist() if sigmas is not None else None

    # ---- POSCAR-aware noise schedule ----
    if poscar_start_sigma is not None and noise > 0:
        if poscar_start_sigma <= 0:
            # Skip entire noise phase — set noise to 0 so loop starts at refine
            noise = 0; total = refine
            noise_active = None
            if sigmas is not None: sigmas = None
        else:
            noise_active = sigmas <= poscar_start_sigma
            # If no steps are active at this sigma, collapse noise to 0
            if not noise_active.any():
                noise = 0; total = refine; noise_active = None; sigmas = None
    else:
        noise_active = torch.ones(noise, dtype=torch.bool, device=device) if noise > 0 else None

    # Convert noise_active to Python list so per-step indexing doesn't force a
    # GPU→CPU sync (matches the sigmas_cpu optimization above).
    noise_active_cpu = noise_active.detach().cpu().tolist() if noise_active is not None else None

    # Extract free_mask once
    free_mask = getattr(data, 'free_mask', None)  # (N, 3) float or None

    # ── RePaint bookkeeping (interface / inpainting) ──
    # A "fixed atom" is one whose free_mask is zero on all three axes.
    # We re-noise exactly those atoms to the scheduled σ each denoise
    # step so the known-region context is in-distribution.  `known_pos`
    # is their pristine target; default to wherever they start.
    fixed_atom = None
    if repaint and free_mask is not None:
        fixed_atom = (free_mask.abs().sum(dim=1) == 0)  # (N,) bool
        if not bool(fixed_atom.any()):
            fixed_atom = None  # nothing pinned → RePaint is a no-op
        elif known_pos is None:
            known_pos = data.pos.detach().clone()

    # ── σ actually carried by the structure entering each step ──
    # Fresh noise of σ_i is injected at the END of step i (disp = model +
    # dx), so the structure entering step i carries the PREVIOUS step's σ.
    # Labeling it with σ_i (the old behaviour) fed DualConv's exp(-σ_t) LR
    # gate a one-step-ahead value — an off-by-one that matters most on the
    # first steps of a steep EDM schedule.  The first denoise step sees
    # either a fully random init (best in-distribution label: σ_top, the
    # training max — training never saw anything noisier) or a
    # near-pristine POSCAR (label the training σ floor).
    if poscar_start_sigma is not None:
        sigma_carry = sig_floor
    elif sigmas_cpu is not None and noise > 0:
        sigma_carry = float(sigmas_cpu[0])
    else:
        sigma_carry = sig_floor

    sa_tensors = [] # Store raw GPU tensors during the loop
    sl = []; t0 = time.time()

    # Pre-flight: validate data.x before touching CUDA
    validate_data(data, model)

    # ── Classifier-free guidance setup (one-time, outside the loop) ──
    # Snapshot every ACTIVE conditioning presence mask once; the per-step
    # guided call zeroes them for the unconditioned forward and restores the
    # snapshot, so no per-step GPU→CPU sync is needed to decide what to
    # toggle.  Presence fields are constant across the trajectory.
    _CFG_PRES = ('energy_present', 'method_present', 'temperature_present',
                 'delta_e_min_present', 'rdf_present', 'rdf_pair_present')
    cfg_weight = float(cfg_weight)
    _cfg_saved = {}
    if cfg_weight != 1.0:
        for _k in _CFG_PRES:
            _v = getattr(data, _k, None)
            if _v is not None and float(_v.abs().sum()) > 0.0:
                _cfg_saved[_k] = _v
        if not _cfg_saved:
            cfg_weight = 1.0   # nothing conditioned — guidance is a no-op
    _cfg_zeros = {k: torch.zeros_like(v) for k, v in _cfg_saved.items()}

    def _model_guided(d):
        """model(d) with classifier-free guidance when cfg_weight != 1."""
        disp_c = model(d)
        if cfg_weight == 1.0:
            return disp_c
        for k, z in _cfg_zeros.items():
            setattr(d, k, z)
        disp_u = model(d)
        for k, v in _cfg_saved.items():
            setattr(d, k, v)
        return disp_u + cfg_weight * (disp_c - disp_u)

    for si in range(total):
        if stop_fn and stop_fn():
            break

        with torch.no_grad():
            if si < noise and noise_active_cpu is not None and not noise_active_cpu[si]:
                pass # Skip inactive noise steps entirely
            else:
                # ── RePaint: re-noise the known/fixed region to the ──
                # ── current σ BEFORE building the neighbor list, so the ──
                # ── edges crossing the interface carry a homogeneous σ. ──
                if fixed_atom is not None:
                    s_known = float(sigmas_cpu[si]) if si < noise \
                        else sig_floor
                    noised = known_pos + torch.randn_like(known_pos) * s_known
                    data.pos[fixed_atom] = noised[fixed_atom]
                    del noised

                vl.update(data)

                # ── Split VL edges into SR + LR each step ──
                if is_dual:
                    _sr_ei, _sr_ea, _lr_ei, _lr_ea = _split_edges_dual(
                        data.edge_index, data.edge_attr, sr_cut_dual)
                    data.edge_index    = _sr_ei
                    data.edge_attr     = _sr_ea
                    data.lr_edge_index = _lr_ei
                    data.lr_edge_attr  = _lr_ea
                    del _sr_ei, _sr_ea, _lr_ei, _lr_ea

                if si < noise:
                    # Query the model at the SCHEDULED σ, not a random
                    # sub-interval down to the floor.  The previous code
                    # used `random.uniform(σ_floor, sigmas[si])`, which
                    # on average queried σ ≈ sigmas[si]/2 — far below
                    # the scheduled noise level.  That mismatches the
                    # training distribution (training queries the model
                    # at exactly the σ that produced the input) and
                    # under-noises the Brownian increment, biasing the
                    # SDE toward whichever attractor the initial random
                    # placement landed near.
                    sigma_step = float(sigmas_cpu[si])
                    data.dx = torch.randn_like(data.pos) * sigma_step
                    # ── B1: expose the CARRIED σ so DualConv scales its LR
                    # branch at the noise level actually present in the
                    # input (see sigma_carry init above), not the σ about
                    # to be injected after this model call.
                    data._sigma_t = torch.tensor(
                        [sigma_carry], dtype=data.pos.dtype, device=data.pos.device)
                    # NOTE: a former "G5" block here added transient σ-scale
                    # noise to fixed atoms' positions around this model call,
                    # intending to give the model a homogeneous σ_t field.
                    # It was a no-op: the model reads only edge_attr/edge_index
                    # (built by vl.update() above, before any position change)
                    # and never data.pos, so the add/restore changed nothing
                    # except leaving a tiny fp residue on atoms meant to stay
                    # fixed.  Removed.  Realising that intent would require
                    # delta-updating edge_attr the way RattleParticles does.
                    disp = _model_guided(data) + data.dx
                    sigma_carry = sigma_step
                else:
                    if hasattr(data, "dx"): del data.dx
                    # Refine phase: clamp σ to the training floor (0.001)
                    # rather than 0 so the LR sigma_t scaling stays inside
                    # the training distribution (the model never saw σ=0
                    # exactly — `RattleParticles` samples from
                    # U(0.001, σ_max)).  At σ=0.001 the LR scale is
                    # exp(-0.001) ≈ 0.999 — effectively full LR weight,
                    # but in-distribution.
                    data._sigma_t = torch.full(
                        (1,), sig_floor,
                        dtype=data.pos.dtype, device=data.pos.device)
                    # G5: at refine σ_floor, transient noise is negligible
                    # (~1e-3 Å) but adding it keeps the inference path
                    # uniform.  Skip for performance — the model is
                    # within-distribution at σ_floor either way.
                    disp = _model_guided(data)

                if free_mask is not None:
                    disp = disp * free_mask
                    # COM detrend over the FREE atoms (same rationale as
                    # the unmasked branch below).  Pinned atoms anchor
                    # only the free atoms inside their receptive field
                    # (num_convs hops × cutoff ≈ 21 Å); free atoms
                    # farther away receive the model's small net-
                    # translation bias unopposed, and over ~10³ steps it
                    # integrates into a coherent drift of the far region
                    # that piles it against the anchored boundary
                    # (observed: a 105 Å a-GeO₂ film translating at a
                    # constant ~0.05 Å/step until it collapsed onto the
                    # substrate; thin films ≤ the receptive field are
                    # fully anchored, which is why small interfaces never
                    # showed it).  A uniform translation is unphysical
                    # under PBC, so subtracting the mean over the free
                    # components removes the spurious mode while leaving
                    # boundary-driven (inhomogeneous) rearrangement
                    # intact.  Fixed atoms stay exactly fixed: the mean
                    # is computed from the masked disp and re-masked.
                    # `com_detrend` selects WHICH axes are detrended (see
                    # docstring): along the stack axis a net displacement
                    # is physical film settling/expansion — detrending it
                    # over-compresses the film (measured 23% vs 75% CN4 on
                    # a quartz/a-GeO₂ interface) — so the default "xy"
                    # detrends only the in-plane axes.
                    _axes = [i for i, ax in enumerate("xyz")
                             if ax in (com_detrend or "")]
                    if _axes:
                        n_free = free_mask.sum(dim=0, keepdim=True)
                        _mean = torch.zeros(
                            (1, 3), dtype=disp.dtype, device=disp.device)
                        _full = disp.sum(dim=0, keepdim=True) / n_free.clamp(min=1)
                        _mean[:, _axes] = _full[:, _axes]
                        disp = (disp - _mean) * free_mask
                else:
                    # COM detrend (free-only case): subtract per-axis mean
                    # displacement to prevent net translational drift over
                    # the long denoise trajectory.  The equivariant model
                    # does not strictly enforce zero-mean output, so over
                    # ~3000 steps small biases accumulate into a multi-Å
                    # COM shift relative to the original frame.
                    disp = disp - disp.mean(dim=0, keepdim=True)

                # H11: NaN guard.  A bad forward (under AMP overflow, or
                # rare numerical edge cases at the σ=σ_min boundary) can
                # silently emit NaN or ±inf displacements; without this
                # check those propagate into data.pos and the next vl.update
                # crashes with a confusing min-image-of-NaN error.
                if not torch.isfinite(disp).all():
                    raise RuntimeError(
                        f"Non-finite displacement at step {si+1}/{total} "
                        f"(σ={sigmas_cpu[si] if si < noise else sig_floor:.4g}). "
                        "Likely causes: AMP overflow, model querying outside "
                        "training σ range, or NaN in input positions.")

                max_d = max_disp_denoise if si < noise else max_disp_refine
                if max_d > 0:
                    norms = disp.norm(dim=1, keepdim=True)
                    scale = torch.clamp(max_d / norms.clamp(min=1e-12), max=1.0)
                    disp = disp * scale
                    del norms, scale

                data.pos.sub_(disp); del disp; _clean(data)

                # Physical excluded-volume floor: forbid unphysical
                # overlaps the score may drive (keeps a dual-cutoff model
                # from collapsing a random init at low σ).
                if min_dist_constraint and min_dist_constraint > 0:
                    data.pos = _enforce_min_dist(
                        data.pos, data.cell, data.pbc, data.numbers,
                        float(min_dist_constraint), device,
                        free_mask=free_mask)

        il = (si == total - 1)
        # Audit 2026-08-06: `snap_every: 0` / `plot_every: 0` in a
        # generation YAML is the natural way to write "never" and used to
        # raise ZeroDivisionError on the FIRST step — after the model load
        # and the initial-structure build, i.e. minutes in.  Nothing
        # validates these keys (`_merge` is a plain dict update).  Treat a
        # non-positive period as "final frame only".  Exactly equivalent
        # to the old expression for every positive value.
        do_snap = (il or (snap_every > 0 and si % snap_every == 0))
        do_plot = (il or (plot_every > 0 and si % plot_every == 0))

        # RePaint: any saved frame should show the substrate at its
        # pristine target, not the σ-floor-noised copy the loop carries.
        if fixed_atom is not None and (do_snap or do_plot):
            data.pos[fixed_atom] = known_pos[fixed_atom]

        raw_pos_np = None
        if do_snap or do_plot:
            # Transfer to CPU and convert to NumPy on the MAIN thread.
            # This takes < 1 ms, guarantees perfect stream synchronization,
            # and prevents empty plots. We use .copy() to ensure the memory
            # is totally detached from PyTorch.
            # Wrap into the PBC unit cell so consumers (on_plot / on_snap
            # callbacks, sa_tensors snapshot list) see in-box coordinates.
            # Previously raw_pos_np was the post-displacement, pre-wrap
            # tensor, and only the final return-value `sa` list was wrapped
            # — intermediate consumers would see atoms outside the box.
            _pos_t = data.pos.detach().cpu()
            _cell_t = data.cell.detach().cpu() if hasattr(data.cell, 'detach') \
                else torch.as_tensor(np.asarray(data.cell), dtype=_pos_t.dtype)
            try:
                _frac = torch.linalg.solve(_cell_t.T, _pos_t.T).T
                _frac = _frac - torch.floor(_frac)
                raw_pos_np = (_frac @ _cell_t).numpy().astype(
                    _pos_t.numpy().dtype).copy()
            except Exception:
                # Degenerate cell or unexpected shape: fall back to the
                # raw (unwrapped) positions rather than crash the snap.
                raw_pos_np = _pos_t.numpy().copy()

        if do_snap:
            ph = "S1" if si < noise else "S2"
            sa_tensors.append(raw_pos_np)
            sl.append("FINAL" if il else f"{ph} step {si+1}")
            if on_snap:
                on_snap(si+1, total, time.time()-t0, raw_pos_np, vl.stats)

        if do_plot and on_plot:
            on_plot(si+1, total, time.time()-t0, raw_pos_np, vl.stats)

    # === LOOP FINISHED ===
    # Now that generation is done, build the final ASE objects
    sa = []
    for pos_array in sa_tensors:
        snap = Atoms(symbols=atoms.get_chemical_symbols(),
                     positions=pos_array,
                     cell=atoms.cell, pbc=True)
        snap.wrap()
        sa.append(snap)

    return sa, sl, time.time()-t0, vl.stats
