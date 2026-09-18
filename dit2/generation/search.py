"""Configuration search by repeated kick-and-quench under the diffusion model.

    from dit2.generation.search import config_search
    res = config_search(model, atoms, device,
                        target_delta_e_min=0.05, target_density=4.2,
                        total_steps=200, n_vol=10, sigma_control=0.3)
    res.trajectory      # list[Atoms], one frame per outer step (+ the input)

WHAT THIS IS.  `generate()` runs ONE reverse-diffusion trajectory: σ walks
monotonically from σ_max to the floor and never goes back up.  That is a
sampler — it draws one structure from the conditioned distribution.  This is a
SEARCH: the structure is repeatedly kicked back up to σ_control and requenched,
so the walker keeps crossing barriers instead of committing to the first basin
it lands in.  Each outer step is

    1.  (every `n_vol` steps) rescale the cell isotropically to the next
        density on the ρ_init → ρ_final ramp, fractional coordinates held
        fixed, so the cell carries the atoms with it;
    2.  kick   — pos += N(0, σ_control), an explicit rattle;
    3.  quench — `generate()` with an EDM sub-schedule from σ_control down to
        the σ floor, then `n_refine` deterministic refine steps;
    4.  record the frame.

Steps 2+3 are SDEdit (Meng et al. 2022, arXiv:2108.01073): noise a real sample
to an intermediate σ, then run the reverse process from that σ.  The kick is
what sets the state at t = σ_control; the reverse SDE's own per-step noise is
the integrator, not a second kick.  They are not redundant.

WHY THE CELL DRIVES THE DENSITY CONDITIONING.  Density is not an independently
settable axis in this package: `mk_data` computes it from the cell it is handed
(`mass / |det(cell)| * 1.66053906660`).  So scaling the cell IS how you condition
on density, and the two can never disagree — `data` is rebuilt after every
rescale so the conditioning always matches the geometry the model is looking at.

CHOOSING σ_control.  This is the one knob that decides whether the run searches
at all.  Too small and each quench returns to the basin it started in (the
trajectory is a fixed point with noise); too large and the kick erases the
structure, so every step is an independent draw from the prior and the "search"
is just repeated cold generation.  σ_control is a DISPLACEMENT IN ÅNGSTRÖM, so
scale it against a bond length, not against σ_max: 0.3 Å is ~20% of a 1.6 Å
Si–O bond — enough to reorder a coordination shell, not enough to dissolve one.
Use `res.rmsd_per_step` and `res.dmin` to tell the two failure modes apart:
a flat rmsd near zero means the kick is too small, an rmsd that saturates at the
box scale means it is too large.  σ_control > the model's training σ_max is
refused outright — the model was never trained there and its score is
meaningless (see `_get_training_sigma_max`).
"""
import math
import time

import numpy as np
import torch
from ase import Atoms

from dit2.generation.core import (generate, mk_data, _get_training_z,
                                 _get_training_sigma_max,
                                 _build_sigma_schedule,
                                 _TRAINING_SIGMA_FLOOR)

# ρ[g/cc] = Σm[amu] / V[Å³] · AMU_PER_A3_TO_G_PER_CC.  Same constant `mk_data`
# uses; keep them equal or the conditioning silently disagrees with the cell.
AMU_PER_A3_TO_G_PER_CC = 1.66053906660


def density_of(atoms) -> float:
    """Mass density (g/cc) of `atoms`, by the same formula `mk_data` uses."""
    vol = abs(float(np.linalg.det(np.array(atoms.cell))))
    if vol <= 0.0:
        raise ValueError(
            f"density_of: cell has non-positive volume ({vol!r}).  A 2-D or "
            f"degenerate cell cannot carry a mass density.")
    return float(atoms.get_masses().sum()) / vol * AMU_PER_A3_TO_G_PER_CC


def scale_cell_to_density(atoms, rho_target: float):
    """Return a copy of `atoms` isotropically rescaled to `rho_target` g/cc.

    Fractional coordinates are invariant (`scale_atoms=True`), so the atoms ride
    with the cell and no bond is stretched relative to the box — the structure
    is compressed or dilated as a whole, which is what a density ramp means.
    Anisotropic cells keep their shape; only the scale changes.
    """
    if not (rho_target > 0.0):
        raise ValueError(f"target density must be > 0, got {rho_target!r}.")
    out = atoms.copy()
    v_now = abs(float(np.linalg.det(np.array(out.cell))))
    v_want = float(out.get_masses().sum()) * AMU_PER_A3_TO_G_PER_CC / rho_target
    s = (v_want / v_now) ** (1.0 / 3.0)
    out.set_cell(np.array(out.cell) * s, scale_atoms=True)
    return out


def _min_dist(atoms, cutoff: float = 3.0) -> float:
    """Shortest interatomic distance under PBC, or inf if none below `cutoff`.

    Cheap collapse detector: a healthy oxide sits near 1.5–1.6 Å, and anything
    under ~1.0 Å means the quench drove atoms through each other.
    """
    from ase.neighborlist import neighbor_list
    try:
        d = neighbor_list('d', atoms, float(cutoff))
    except Exception:
        return float('nan')
    return float(d.min()) if len(d) else float('inf')


def _rmsd(a, b) -> float:
    """Min-image RMSD between two frames SHARING a cell (Å).

    Used for the per-step value, where the rescale has already happened before
    the `prev` snapshot is taken, so both frames sit in the same box.
    """
    cell = np.array(a.cell)
    try:
        inv = np.linalg.inv(cell)
    except np.linalg.LinAlgError:
        return float('nan')
    df = (b.positions - a.positions) @ inv
    df -= np.round(df)                      # min image
    d = df @ cell
    return float(np.sqrt((d ** 2).sum(axis=1).mean()))


def _rmsd_frac(a, b) -> float:
    """RMSD between frames whose CELLS DIFFER, measured in b's box (Å).

    Under a density ramp the start and current frames live in different cells,
    so a Cartesian difference double-counts the uniform compression that the
    ramp applied on purpose.  Comparing FRACTIONAL coordinates and mapping the
    difference through the current cell reports only the rearrangement — a
    structure that was merely squeezed, atom for atom, scores 0.
    """
    try:
        fa = a.get_scaled_positions(wrap=False)
        fb = b.get_scaled_positions(wrap=False)
    except Exception:
        return float('nan')
    df = fb - fa
    df -= np.round(df)                      # min image
    d = df @ np.array(b.cell)
    return float(np.sqrt((d ** 2).sum(axis=1).mean()))


class SearchResult:
    """Trajectory plus the per-step diagnostics needed to tune σ_control.

    Attributes
    ----------
    trajectory : list[Atoms]
        Frame 0 is the (rescaled) input; frame k is the structure after outer
        step k.  Cells DIFFER between frames whenever a density ramp is active,
        so write these with a format that stores a per-frame cell (extxyz), not
        one that assumes a fixed cell (XDATCAR).
    density, dmin, rmsd_per_step, rmsd_from_start : list[float]
    sigma_control_used : list[float]
    elapsed : float
    """

    def __init__(self, trajectory, density, dmin, rmsd_per_step,
                 rmsd_from_start, sigma_control_used, elapsed, meta):
        self.trajectory = trajectory
        self.density = density
        self.dmin = dmin
        self.rmsd_per_step = rmsd_per_step
        self.rmsd_from_start = rmsd_from_start
        self.sigma_control_used = sigma_control_used
        self.elapsed = elapsed
        self.meta = meta

    def __len__(self):
        return len(self.trajectory)

    @property
    def final(self):
        return self.trajectory[-1]

    def write(self, path):
        """Write the whole trajectory to `path` (extxyz keeps per-frame cells)."""
        from ase.io import write as ase_write
        ase_write(str(path), self.trajectory, format='extxyz')
        return path

    def summary(self) -> str:
        # Count OUTER STEPS, not stored frames -- `keep_every` decouples them,
        # and reporting frames here made a keep_every=3 run of 12 steps print
        # "4 outer steps" at 3x the true cost per step.
        n = len(self.density) - 1
        collapsed = sum(1 for d in self.dmin if np.isfinite(d) and d < 1.0)
        return (f"config_search: {n} outer steps in {self.elapsed:.1f}s "
                f"({self.elapsed / max(n, 1):.2f}s/step)\n"
                f"  density   {self.density[0]:.4f} -> {self.density[-1]:.4f} g/cc\n"
                f"  dmin      min {np.nanmin(self.dmin):.3f} Å, "
                f"final {self.dmin[-1]:.3f} Å, {collapsed} frame(s) < 1.0 Å\n"
                f"  rmsd/step mean {np.nanmean(self.rmsd_per_step):.3f} Å, "
                f"from start {self.rmsd_from_start[-1]:.3f} Å")


def config_search(model, atoms, device, *,
                  target_delta_e_min=None,
                  target_energy=None,
                  target_density=None,
                  total_steps=100,
                  n_vol=10,
                  sigma_control=0.3,
                  sigma_control_final=None,
                  n_denoise=12,
                  n_refine=8,
                  gnn_cut='auto',
                  vskin=1.0,
                  min_dist_constraint=0.0,
                  refine_sigma=None,
                  sigma_schedule="edm",
                  sigma_rho=7.0,
                  cfg_weight=1.0,
                  cond_weights=None,
                  method=None,
                  temperature=None,
                  max_disp_denoise=0.0,
                  max_disp_refine=0.0,
                  com_detrend="xyz",
                  keep_every=1,
                  seed=None,
                  on_step=None,
                  stop_fn=None,
                  track_dmin=True,
                  collapse_dmin=0.0,
                  all_model_elements=None,
                  verbose=True):
    """Search configuration space by kicking and requenching under the model.

    Parameters
    ----------
    model, atoms, device
        A loaded model, the INITIAL structure (ASE Atoms with a cell and pbc),
        and the torch device.  `atoms` is never mutated.
    target_delta_e_min : float or None
        ΔE conditioning (eV/atom above the composition's floor) — the "energy
        conditioning" axis.  PASS IT.  Leaving it None generates unconditioned
        on ΔE, which measurably flattens the conditioning response; the standing
        rule in this project is to always supply it at generation.  A warning is
        emitted if it is None and the model has a ΔE head.
    target_energy : float or None
        The legacy absolute-energy axis (`data.energy`), independent of ΔE.
        None = unconditioned on it.
    target_density : float or None
        FINAL mass density (g/cc).  The cell is ramped linearly from the input
        structure's density to this value.  None = hold the input density (no
        rescale ever happens, and `n_vol` is irrelevant).
    total_steps : int
        Number of outer kick-and-quench steps.  Model calls ≈
        `total_steps * (n_denoise + n_refine)`, which is what actually costs.
    n_vol : int
        Rescale the cell every `n_vol` outer steps.  The ramp is indexed by the
        UPDATE, not the step, so the last update lands exactly on
        `target_density` — a ramp parameterised by step/(total_steps-1) and
        applied only at multiples of n_vol stops short of the target and
        silently under-densifies.
    sigma_control : float
        Kick amplitude in Å (per Cartesian component).  See the module
        docstring — this is the knob that decides whether the run searches.
        Must be ≤ the model's training σ_max.
    sigma_control_final : float or None
        Anneal the kick linearly from `sigma_control` to this over the run
        (simulated-annealing style: search wide, then settle).  None = constant.
    n_denoise, n_refine : int
        Inner quench: `n_denoise` stochastic steps on an EDM schedule from
        σ_control down to the floor, then `n_refine` deterministic refine steps.
        n_denoise=0 gives a pure deterministic quench (kick, then relax).
    min_dist_constraint : float
        Excluded-volume floor (Å) passed to `generate`.  Strongly advised when
        ramping to high density: compression is exactly the regime where the
        score drives overlaps.  Use `resolve_min_dist_constraint(...)` from
        `dit2.generation.interface` to get the model-appropriate default.
    refine_sigma : float or None
        σ the quench TERMINATES at (dit6 only; ignored by packages whose
        `generate` has no such parameter).  The low-σ regime is near-blind on
        some checkpoints, so terminating at ~0.02 rather than 0.001 can be
        worth a large energy gain — but it is a property of the checkpoint and
        must be measured per model, not assumed.
    gnn_cut : float or 'auto'
        Neighbour-list cutoff (Å).  'auto' (default) reads the cutoff the model
        was TRAINED with from its own metadata -- do not guess it.  Passing a
        cutoff smaller than the trained one silently truncates every atom's
        neighbourhood, so the model is evaluated on graphs it never saw and the
        quench is wrong in a way nothing downstream reports.  (Dual-cutoff
        models override this internally with their long-range radius.)
    keep_every : int
        Store every k-th frame (the final frame is always stored).  Frames are
        full structures; a 1220-atom 5000-step run is not something you want
        entirely in RAM.
    collapse_dmin : float
        Abort the search if the shortest interatomic distance falls below this
        (Å).  0 = never abort.  Requires `track_dmin`.
    on_step : callable(step, total, atoms, info) or None
    stop_fn : callable() -> bool or None
        Cooperative interrupt, polled once per outer step.

    Returns
    -------
    SearchResult
    """
    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed) % (2 ** 32))

    total_steps = int(total_steps)
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}.")
    n_vol = int(n_vol)
    if n_vol < 1:
        raise ValueError(
            f"n_vol must be >= 1 (it is a step period, not a fraction), "
            f"got {n_vol}.")
    if n_denoise < 0 or n_refine < 0 or (n_denoise + n_refine) < 1:
        raise ValueError(
            f"the inner quench needs at least one step: got "
            f"n_denoise={n_denoise}, n_refine={n_refine}.")

    # σ_control above the training σ_max is not a "stronger kick", it is a
    # query the model has no score for -- refuse rather than silently produce
    # noise-shaped output.
    sigma_top = float(_get_training_sigma_max(model))
    for _name, _val in (("sigma_control", sigma_control),
                        ("sigma_control_final", sigma_control_final)):
        if _val is None:
            continue
        if not (float(_val) > 0.0):
            raise ValueError(f"{_name} must be > 0, got {_val!r}.")
        if float(_val) > sigma_top:
            raise ValueError(
                f"{_name}={_val} exceeds the model's training σ_max "
                f"({sigma_top}).  The model never saw noise that large, so its "
                f"score there is extrapolation and the quench will not recover "
                f"a physical structure.  Lower the kick, or retrain with a "
                f"larger σ_max.")

    # Resolve the cutoff from the checkpoint rather than trusting a default.
    # A too-small cutoff is the quietest way to get wrong answers here: the
    # neighbour list simply omits edges and every subsequent number looks
    # plausible.
    if isinstance(gnn_cut, str):
        if gnn_cut.strip().lower() != 'auto':
            raise ValueError(
                f"gnn_cut must be a number or 'auto', got {gnn_cut!r}.")
        from dit2.compat import get_mi as _get_mi
        _mi = _get_mi(model)
        _c = _mi.get("GNN cutoff (A)")
        if _c is None:
            raise ValueError(
                "gnn_cut='auto' but the model exposes no 'GNN cutoff (A)' in "
                "its metadata.  Pass the cutoff the model was trained with "
                "explicitly -- guessing it silently truncates the graph.")
        gnn_cut = float(_c)
        if verbose:
            print(f"config_search: gnn_cut='auto' -> {gnn_cut} Å "
                  f"(from the checkpoint)")

    if all_model_elements is None:
        try:
            from dit2.compat import get_model_elements
            all_model_elements = get_model_elements(model)
        except Exception:
            all_model_elements = []

    if target_delta_e_min is None and verbose:
        print("config_search: WARNING — target_delta_e_min is None, so the run "
              "is UNCONDITIONED on ΔE.  On this project's models that flattens "
              "the conditioning response; pass the ΔE you actually want.")

    start = atoms.copy()
    rho_init = density_of(start)
    rho_final = rho_init if target_density is None else float(target_density)

    # Ramp indexed by UPDATE so the final update hits rho_final exactly.
    n_updates = int(math.ceil(total_steps / float(n_vol)))
    n_updates = max(n_updates, 1)

    # n_vol >= total_steps collapses the ramp to a SINGLE update, which jumps
    # straight to rho_final at step 0 -- a legitimate mode ("just resize, then
    # search") but never what someone asking for a gradual ramp meant.
    if (verbose and target_density is not None and n_updates == 1
            and abs(rho_final - rho_init) > 1e-9):
        print(f"config_search: WARNING — n_vol={n_vol} >= total_steps="
              f"{total_steps}, so the density goes {rho_init:.4f} -> "
              f"{rho_final:.4f} g/cc in ONE jump at step 0, not as a ramp. "
              f"Lower n_vol for a gradual ramp.")

    def rho_at_update(k):
        if n_updates == 1:
            return rho_final
        return rho_init + (rho_final - rho_init) * (k / float(n_updates - 1))

    def sigma_at_step(i):
        if sigma_control_final is None or total_steps == 1:
            return float(sigma_control)
        f = i / float(total_steps - 1)
        return float(sigma_control) + (float(sigma_control_final)
                                       - float(sigma_control)) * f

    # dit6's generate accepts refine_sigma; dit2's does not.  Detect rather
    # than fork the file, so the same body is correct in both packages.
    import inspect
    _gen_params = inspect.signature(generate).parameters
    _extra = {}
    if refine_sigma is not None:
        if 'refine_sigma' in _gen_params:
            _extra['refine_sigma'] = refine_sigma
        else:
            raise TypeError(
                "refine_sigma was requested but this package's generate() does "
                "not support it (dit2 terminates at the training σ floor). "
                "Drop the argument or run the search in dit6.")
    sig_floor = float(refine_sigma) if refine_sigma is not None \
        else _TRAINING_SIGMA_FLOOR

    training_z = _get_training_z(model)
    cur = scale_cell_to_density(start, rho_at_update(0)) \
        if target_density is not None else start.copy()

    traj = [cur.copy()]
    dens = [density_of(cur)]
    dmin = [_min_dist(cur) if track_dmin else float('nan')]
    rmsd_step, rmsd_start, sig_used = [0.0], [0.0], [float('nan')]
    traj[0].info.update(step=0, density=dens[0], phase='input')

    t0 = time.time()
    aborted = None
    appended_last = True          # frame 0 is stored
    for i in range(total_steps):
        if stop_fn is not None and stop_fn():
            aborted = 'stop_fn'
            break

        # ── 1. density ramp (fractional coordinates preserved) ──
        if target_density is not None and i % n_vol == 0:
            cur = scale_cell_to_density(cur, rho_at_update(i // n_vol))

        prev = cur.copy()
        s_ctl = sigma_at_step(i)

        # ── 2. kick ──
        cur.positions = cur.positions + np.random.randn(
            len(cur), 3) * s_ctl

        # ── 3. quench ──
        # data is rebuilt every step: the cell may have just changed, and
        # `mk_data` derives the density conditioning from it.
        data = mk_data(cur, all_model_elements, target_energy, device,
                       training_z=training_z,
                       method=method, temperature=temperature,
                       delta_e_min=target_delta_e_min,
                       cond_weights=cond_weights)
        sigmas = None
        if n_denoise > 0:
            sigmas = _build_sigma_schedule(
                s_ctl, sig_floor, n_denoise,
                mode=sigma_schedule, rho=sigma_rho, device=device)
        sa, _sl, _el, _vs = generate(
            model, data, cur, n_denoise, n_refine,
            0, 0,                       # snap_every / plot_every: final only
            gnn_cut, vskin, device,
            sigmas=sigmas,
            min_dist_constraint=min_dist_constraint,
            max_disp_denoise=max_disp_denoise,
            max_disp_refine=max_disp_refine,
            sigma_schedule=sigma_schedule, sigma_rho=sigma_rho,
            com_detrend=com_detrend, cfg_weight=cfg_weight,
            **_extra)
        del data
        if not sa:
            aborted = 'generate returned no frame'
            break
        cur = sa[-1]

        # ── 4. record ──
        d = _min_dist(cur) if track_dmin else float('nan')
        info = dict(step=i + 1, density=density_of(cur), dmin=d,
                    sigma_control=s_ctl,
                    delta_e_min=target_delta_e_min,
                    rmsd_step=_rmsd(prev, cur))
        dens.append(info['density'])
        dmin.append(d)
        rmsd_step.append(info['rmsd_step'])
        rmsd_start.append(_rmsd_frac(traj[0], cur))
        sig_used.append(s_ctl)
        appended_last = False
        if (i + 1) % max(int(keep_every), 1) == 0 or i == total_steps - 1:
            frame = cur.copy()
            frame.info.update(info)
            traj.append(frame)
            appended_last = True
        if on_step is not None:
            on_step(i + 1, total_steps, cur, info)
        if verbose and ((i + 1) % max(total_steps // 20, 1) == 0):
            print(f"  step {i+1:>5}/{total_steps}  ρ={info['density']:.4f}  "
                  f"σ_ctl={s_ctl:.3f}  dmin={d:.3f}  "
                  f"rmsd={info['rmsd_step']:.3f}", flush=True)

        if collapse_dmin and track_dmin and np.isfinite(d) and d < collapse_dmin:
            aborted = (f"collapse: dmin {d:.3f} Å < collapse_dmin "
                       f"{collapse_dmin} Å at step {i+1}")
            break

    elapsed = time.time() - t0
    # `keep_every` may have skipped the frame the run actually ended on (an
    # abort can land anywhere), and the LAST structure is the one the caller
    # came for.  Append it unless it is already the stored tail.
    if not appended_last and len(dens) > 1:
        f = cur.copy()
        f.info.update(step=len(dens) - 1, density=dens[-1], dmin=dmin[-1])
        traj.append(f)

    meta = dict(rho_init=rho_init, rho_final=rho_final, n_updates=n_updates,
                n_vol=n_vol, total_steps=total_steps,
                sigma_control=sigma_control,
                sigma_control_final=sigma_control_final,
                n_denoise=n_denoise, n_refine=n_refine,
                sigma_floor=sig_floor, sigma_max=sigma_top,
                target_delta_e_min=target_delta_e_min,
                target_energy=target_energy,
                min_dist_constraint=min_dist_constraint,
                model_calls=len(dens[1:]) * (n_denoise + n_refine),
                aborted=aborted)
    res = SearchResult(traj, dens, dmin, rmsd_step, rmsd_start, sig_used,
                       elapsed, meta)
    if verbose:
        print(res.summary())
        if aborted:
            print(f"  ABORTED: {aborted}")
    return res
