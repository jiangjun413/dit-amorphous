"""Two-level-system (double-well) identification from a kick-and-quench run.

    from dit2.generation.search import config_search
    from dit2.generation.tls import find_double_wells, tunneling_parameter

    res  = config_search(model, atoms, dev, sigma_control=0.3, total_steps=400)
    cand = find_double_wells(res.trajectory)          # geometry only
    cand = find_double_wells(res.trajectory, energies=E)   # + asymmetry

WHAT THIS DOES.  `config_search` visits a sequence of quenched configurations.
Most consecutive frames fall back into the basin they started in; occasionally
the walker crosses a barrier and lands somewhere else.  This module groups the
frames into distinct basins and reports the pairs that look like a TLS: two
configurations that are CLOSE in configuration space, differ by a LOCALIZED
rearrangement of a few atoms, and (given energies) are nearly degenerate.

WHAT THIS DOES NOT DO, AND CANNOT.
1.  It does not prove the frames are minima.  A quench lands near a mode of the
    LEARNED distribution, not at a force-zero point of the true PES.  Every
    candidate must be relaxed (DFT or an MLIP) before it is called a well; use
    `write_relax_inputs` and re-run this on the relaxed pair.
2.  dit2 has NO energy head, so asymmetry and barrier CANNOT come from the
    model.  `energies` is an argument, never an estimate.  Passing the
    denoiser score as a pseudo-energy would be wrong: it is a gradient of a
    log-density under the model, not a potential energy, and at the refine
    sigma this package generates at it is measurably ill-calibrated.
3.  The barrier needs a path.  `interpolate_path` emits images for an NEB or a
    single-point scan; nothing here invents a saddle.

CONVENTIONS.
*  Displacements are minimum-image via `ase.geometry.find_mic`, which is exact
   for a general triclinic cell.  `search._rmsd` uses fractional rounding,
   which is exact only for cells that are not strongly skewed; TLS candidates
   are selected on a sub-Angstrom threshold, so the general routine is used
   here deliberately.
*  Net translation is removed (mass-weighted) before any metric.  `generate()`
   detrends the centre of mass in xy only by default (`com_detrend="xy"`), so a
   rigid z drift survives the search and would otherwise register as a large
   fake displacement on every atom.
*  Frames whose CELLS differ are refused.  A density ramp rescales the cell
   between frames, and a displacement across two different boxes is not a
   physical displacement.  Run the search at fixed density for TLS work.

UNITS.  Lengths Angstrom, energies eV, masses amu.  The mass-weighted distance
`d_mw` is in sqrt(amu)*Angstrom, which is what the tunneling integral wants.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence, List

import numpy as np
from ase.geometry import find_mic

# CODATA 2018, in the unit system above.
_HBAR_JS = 1.054571817e-34
_AMU_KG = 1.66053906660e-27
_EV_J = 1.602176634e-19
_ANG_M = 1.0e-10

__all__ = ["PairMetrics", "TLSCandidate", "pair_displacement", "pair_metrics",
           "cluster_basins", "find_double_wells", "tunneling_parameter",
           "tunneling_splitting", "interpolate_path", "write_report",
           "write_relax_inputs"]


def _check_comparable(a, b, cell_tol=1e-6):
    """Refuse pairs whose displacement would be meaningless."""
    if len(a) != len(b):
        raise ValueError(f"atom count differs: {len(a)} vs {len(b)}")
    sa, sb = a.get_chemical_symbols(), b.get_chemical_symbols()
    if sa != sb:
        n = sum(1 for x, y in zip(sa, sb) if x != y)
        raise ValueError(
            f"species order differs in {n} position(s); this module assumes "
            "atom identity is preserved (config_search does preserve it -- a "
            "sorted or re-read structure may not)")
    ca, cb = np.asarray(a.cell), np.asarray(b.cell)
    if not np.allclose(ca, cb, atol=cell_tol, rtol=0):
        raise ValueError(
            "cells differ between frames -- a displacement across two boxes is "
            "not physical. A density ramp (n_vol) rescales the cell; re-run "
            "config_search with target_density=None for TLS work, or slice the "
            "trajectory to a constant-cell segment.")


def pair_displacement(a, b, remove_com=True):
    """Per-atom minimum-image displacement b-a, shape (N,3), Angstrom."""
    _check_comparable(a, b)
    raw = b.positions - a.positions
    d, _ = find_mic(raw, np.asarray(a.cell), pbc=a.pbc)
    if remove_com:
        m = a.get_masses()[:, None]
        d = d - (m * d).sum(axis=0) / m.sum()
    return d


@dataclass
class PairMetrics:
    """Geometry of one candidate transition. Lengths Angstrom."""
    d_rms: float          # sqrt(mean |dr|^2) -- per-atom scale
    d_total: float        # sqrt(sum  |dr|^2) -- configuration-space distance
    d_mw: float           # sqrt(sum m|dr|^2) over ALL atoms, sqrt(amu)*Ang
    d_mw_active: float    # same, restricted to the active atoms
    rattle_fraction: float  # 1 - (d_mw_active/d_mw)^2, share of d_mw^2 that
                            # is background motion rather than the hop
    max_disp: float
    n_active: int         # atoms moving more than `active_thresh`
    participation: float  # PR in (0,1]; N*PR ~ number of atoms involved
    n_participating: float
    active_indices: np.ndarray = field(repr=False)

    def as_row(self):
        d = asdict(self)
        d.pop("active_indices")
        return d


def pair_metrics(a, b, active_thresh=0.25, remove_com=True) -> PairMetrics:
    """Displacement, localization and mass weighting for one pair.

    active_thresh : Angstrom
        An atom counts as active above this. 0.25 A is well past thermal
        rattle at room temperature yet well below a bond length, so it
        separates "this atom moved" from "the network breathed".
    participation : (sum u^2)^2 / (N * sum u^4), the standard participation
        ratio. It is 1 for a uniform displacement and ~n/N when n atoms carry
        the motion, so N*PR estimates how many atoms are involved WITHOUT
        needing a threshold -- report it alongside n_active, which does.
    """
    d = pair_displacement(a, b, remove_com=remove_com)
    u2 = (d ** 2).sum(axis=1)
    n = len(a)
    s2, s4 = u2.sum(), (u2 ** 2).sum()
    pr = float(s2 ** 2 / (n * s4)) if s4 > 0 else 0.0
    act = np.where(np.sqrt(u2) > active_thresh)[0]
    mass = a.get_masses()
    d_mw_all = float(np.sqrt((mass * u2).sum()))
    d_mw_act = float(np.sqrt((mass[act] * u2[act]).sum())) if act.size else 0.0
    return PairMetrics(
        d_rms=float(np.sqrt(u2.mean())),
        d_total=float(np.sqrt(s2)),
        d_mw=d_mw_all,
        d_mw_active=d_mw_act,
        rattle_fraction=(1.0 - (d_mw_act / d_mw_all) ** 2) if d_mw_all > 0 else 0.0,
        max_disp=float(np.sqrt(u2.max())) if n else 0.0,
        n_active=int(act.size),
        participation=pr,
        n_participating=float(n * pr),
        active_indices=act,
    )


def cluster_basins(frames, tol=0.30, active_thresh=0.25, remove_com=True):
    """Group frames into basins; two frames match if d_rms < tol (Angstrom).

    Greedy against a fixed representative per basin rather than single-linkage:
    single-linkage chains through a slowly drifting trajectory and merges the
    whole run into one basin. Returns (labels, representatives).

    tol should sit between the intra-basin rattle and the inter-basin
    displacement. Read it off `res.rmsd_per_step`: the distribution is bimodal
    when the search is working, and tol belongs in the gap.
    """
    labels = np.full(len(frames), -1, dtype=int)
    reps: List[int] = []
    for i, f in enumerate(frames):
        for b, r in enumerate(reps):
            if pair_metrics(frames[r], f, active_thresh, remove_com).d_rms < tol:
                labels[i] = b
                break
        else:
            labels[i] = len(reps)
            reps.append(i)
    return labels, reps


@dataclass
class TLSCandidate:
    basin_a: int
    basin_b: int
    frame_a: int
    frame_b: int
    n_visits_a: int
    n_visits_b: int
    metrics: PairMetrics
    asymmetry_eV: Optional[float] = None      # |E_b - E_a|, external energies
    barrier_eV: Optional[float] = None        # external; never from the model
    lam: Optional[float] = None               # WKB tunneling parameter
    splitting_eV: Optional[float] = None

    def as_row(self):
        r = {"basin_a": self.basin_a, "basin_b": self.basin_b,
             "frame_a": self.frame_a, "frame_b": self.frame_b,
             "n_visits_a": self.n_visits_a, "n_visits_b": self.n_visits_b}
        r.update(self.metrics.as_row())
        r.update({"asymmetry_eV": self.asymmetry_eV,
                  "barrier_eV": self.barrier_eV,
                  "lam": self.lam, "splitting_eV": self.splitting_eV})
        return r


def tunneling_parameter(d_mw, barrier_eV):
    """WKB tunneling parameter lambda for a symmetric double well.

    lambda = (d_mw / 2 hbar) * sqrt(2 V), evaluated in MASS-WEIGHTED
    coordinates, where the effective mass is 1 by construction, so no separate
    m_eff has to be guessed. d_mw in sqrt(amu)*Angstrom, V in eV.

    The tunnel splitting is Delta0 = hbar*Omega*exp(-lambda), so lambda is the
    quantity that decides whether a double well is a TLS at all: the standard
    tunneling model keeps roughly 1 < lambda < 20, below which the two wells
    are one state and above which the system never tunnels on lab timescales.
    """
    d_si = np.asarray(d_mw, float) * np.sqrt(_AMU_KG) * _ANG_M
    v_si = np.asarray(barrier_eV, float) * _EV_J
    with np.errstate(invalid="ignore"):
        return d_si * np.sqrt(2.0 * v_si) / (2.0 * _HBAR_JS)


def tunneling_splitting(d_mw, barrier_eV, omega_eV=0.010):
    """Delta0 = hbar*Omega*exp(-lambda), in eV.

    omega_eV is the attempt/zero-point energy hbar*Omega of the well. 10 meV
    (~80 cm^-1) is the usual order for the soft modes that carry TLS in oxide
    glasses; it is a PREFACTOR ASSUMPTION, not a measurement, and it multiplies
    the answer linearly -- state it whenever you quote a splitting.
    """
    return omega_eV * np.exp(-tunneling_parameter(d_mw, barrier_eV))


def find_double_wells(frames, energies=None, tol=0.30, active_thresh=0.25,
                      d_rms_range=(0.05, 1.0), max_active=None,
                      max_participating=None, asym_max=None,
                      barriers=None, omega_eV=0.010, remove_com=True):
    """Identify double-well (TLS candidate) pairs among quenched frames.

    energies : per-FRAME energies (eV/cell) from DFT or an MLIP, or None.
        Never from the model. With energies, each basin is represented by its
        LOWEST frame and `asymmetry_eV` is the gap between those.
    d_rms_range : (lo, hi) Angstrom -- distinct but still nearby. `lo` rejects
        the same basin re-found, `hi` rejects wholesale reconstructions, which
        are not two-level anything.
    max_active / max_participating : localization cuts. A TLS is a few atoms
        moving, not the network. Leave None to rank rather than filter.
    barriers : dict {(basin_a, basin_b): V_eV} from an NEB, or None. Only pairs
        with a barrier get `lam` and `splitting_eV`.
    """
    labels, reps = cluster_basins(frames, tol, active_thresh, remove_com)
    nb = len(reps)
    if energies is not None:
        energies = np.asarray(energies, float)
        if energies.shape != (len(frames),):
            raise ValueError(
                f"energies has shape {energies.shape}, expected ({len(frames)},)"
                " -- one per FRAME of the trajectory")
        rep_of = []
        for b in range(nb):
            idx = np.where(labels == b)[0]
            rep_of.append(int(idx[np.argmin(energies[idx])]))
    else:
        rep_of = list(reps)

    lo, hi = d_rms_range
    out = []
    for i in range(nb):
        for j in range(i + 1, nb):
            fa, fb = rep_of[i], rep_of[j]
            m = pair_metrics(frames[fa], frames[fb], active_thresh, remove_com)
            if not (lo <= m.d_rms <= hi):
                continue
            if max_active is not None and m.n_active > max_active:
                continue
            if max_participating is not None and m.n_participating > max_participating:
                continue
            asym = None
            if energies is not None:
                asym = float(abs(energies[fb] - energies[fa]))
                if asym_max is not None and asym > asym_max:
                    continue
            V = None if barriers is None else barriers.get((i, j), barriers.get((j, i)))
            lam = sp = None
            if V is not None:
                # d_mw_active, NOT d_mw: see "WHICH DISTANCE FEEDS LAMBDA".
                lam = float(tunneling_parameter(m.d_mw_active, V))
                sp = float(tunneling_splitting(m.d_mw_active, V, omega_eV))
            out.append(TLSCandidate(
                basin_a=i, basin_b=j, frame_a=fa, frame_b=fb,
                n_visits_a=int((labels == i).sum()),
                n_visits_b=int((labels == j).sum()),
                metrics=m, asymmetry_eV=asym, barrier_eV=V,
                lam=lam, splitting_eV=sp))

    # most TLS-like first: localized, then near-degenerate, then short-hop
    out.sort(key=lambda c: (c.metrics.n_participating,
                            c.asymmetry_eV if c.asymmetry_eV is not None else 0.0,
                            c.metrics.d_rms))
    return out


def interpolate_path(a, b, n_images=7, remove_com=True):
    """Linear min-image images from a to b, INCLUSIVE, for an NEB or a scan.

    Straight-line interpolation gives an UPPER bound on the barrier; it is a
    starting path for an NEB, not a barrier. Feed the interior images to the
    NEB and take the converged saddle.
    """
    if n_images < 2:
        raise ValueError("n_images must be >= 2 (both endpoints are included)")
    d = pair_displacement(a, b, remove_com=remove_com)
    out = []
    for k in range(n_images):
        img = a.copy()
        img.positions = a.positions + d * (k / (n_images - 1))
        out.append(img)
    return out


def write_report(candidates, path):
    """One CSV row per candidate."""
    import csv
    rows = [c.as_row() for c in candidates]
    cols = (list(rows[0].keys()) if rows else
            ["basin_a", "basin_b", "frame_a", "frame_b", "d_rms", "d_mw"])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path


def write_relax_inputs(candidates, frames, outdir, fmt="vasp"):
    """Write both endpoints of each candidate for external relaxation.

    Step 1 of the workflow this module cannot do itself: these are denoised
    structures, and a pair is only a double well once both ends relax to
    distinct force-zero minima. Re-run find_double_wells on the relaxed pair.
    """
    import os
    from ase.io import write as ase_write
    os.makedirs(outdir, exist_ok=True)
    made = []
    for k, c in enumerate(candidates):
        for tag, fi in (("A", c.frame_a), ("B", c.frame_b)):
            d = os.path.join(outdir, f"tls{k:03d}_b{c.basin_a}_{c.basin_b}_{tag}")
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, "POSCAR" if fmt == "vasp" else f"{tag}.xyz")
            ase_write(p, frames[fi], format=fmt)
            made.append(p)
    return made
