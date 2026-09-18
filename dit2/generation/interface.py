"""Interface / heterostructure generation for the DiT diffusion model.

The bulk generator (`dit.generation.core.generate`) samples a single
homogeneous amorphous cell conditioned on one global scalar (energy /
density).  An *interface* is a two-phase system — a substrate + an
amorphous film, an amorphous/amorphous heterojunction, or a
crystal/amorphous boundary — where the two sides differ in composition,
density, and local order, separated by a boundary region that is neither
bulk phase.

This module extends the model to that setting **without changing the
learned score**, by combining three ingredients that already exist in
the codebase:

1.  A slab *initial-structure builder* (`build_interface_structure`)
    that stacks two regions along one lattice axis into a single
    periodic cell.  Because the cell is periodic along the stacking
    axis, an A|B stack yields two equivalent interfaces (A/B and, across
    the periodic boundary, B/A) — the standard slab geometry for
    interface molecular dynamics.

2.  The existing per-atom ``free_mask`` primitive, used to pin a
    substrate region so only the film + boundary atoms are denoised
    (inpainting).

3.  RePaint conditioning (`generate(..., repaint=True)`): the pinned
    substrate is re-noised to the scheduled σ at every reverse step so
    the boundary edges the model sees are in-distribution, then clamped
    back to its pristine positions for the output.

For higher fidelity than a bulk model can give, the model also supports
optional per-atom *region conditioning* (``n_regions`` /
``RegionEmbed``); a region-aware checkpoint fine-tuned on interface data
consumes the ``region`` labels this builder produces.  A plain bulk
checkpoint simply ignores them, so the same builder feeds both paths.
"""

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers as ASE_Z

from dit2.utils.physics import _AMU_TO_GRAM


_AXIS = {"x": 0, "y": 1, "z": 2, 0: 0, 1: 1, 2: 2}

# Atom-count above which a dual-cutoff (long-range) model becomes unsafe
# for the `start_sigma` relaxation path (the low-σ-on-random-init trick
# used to keep the two phases from mixing).  At low σ the model expects a
# nearly-finished glass, but the interface init is a *random* placement —
# out of distribution.  A short-range model shrugs this off (bounded
# ~3–4 Å corrections at any σ), but the dual model's long-range branch
# emits a pathological tail (tens–hundreds of Å for a few atoms) that
# cascades into collapse once the system is large enough to seed it.
# Measured on model_dualcut7 (random 2400-atom init): max |disp| at σ=0.3
# is 3.6 Å (graphite) vs 170 Å (dual).  A FULL anneal from σ_max keeps
# the dual model in-distribution (high-σ ≈ random) and is stable — which
# is why the same model generates fine bulk structures — but a full anneal
# blends the phases, so it isn't a drop-in interface fix.
_DUAL_CUTOFF_SAFE_ATOMS = 400


def _model_conv_type(model):
    """Walk the .model wrapper chain (adapter / sharded) to the conv_type."""
    inner = model
    for _ in range(4):
        ct = getattr(inner, "conv_type", None)
        if ct is not None:
            return ct
        nxt = getattr(inner, "model", None)
        if nxt is None or nxt is inner:
            break
        inner = nxt
    return getattr(inner, "conv_type", None)


# Default excluded-volume floor (Å) auto-enabled for long-range models —
# below the shortest real oxide bond (Si-O ≈ 1.6 Å), so it only removes
# unphysical overlaps.
_DUAL_CUTOFF_MIN_DIST = 1.1


def resolve_min_dist_constraint(value, model, dual_default=_DUAL_CUTOFF_MIN_DIST):
    """Resolve a ``min_dist_constraint`` config value, auto-enabling the
    excluded-volume floor for dual-cutoff (long-range) models.

    Returns ``(resolved_float, was_auto)``.

    * ``None`` or ``"auto"`` → `dual_default` for a dual-cutoff model
      (which otherwise collapses a random init at low σ), else 0.0.
    * a number → used verbatim (explicit; ``0`` disables even for a
      long-range model).
    """
    if value is None or (isinstance(value, str)
                         and value.strip().lower() == "auto"):
        is_dual = _model_conv_type(model) == "dual_cutoff"
        return (float(dual_default) if is_dual else 0.0), True
    return float(value), False


def interface_stability_warning(model, n_atoms, start_sigma=None):
    """Return a warning string if this (model, size, schedule) combo is
    known to be unstable for interface generation, else None.

    Flags dual-cutoff models run with the `start_sigma` relaxation path
    (start_sigma set and < the training σ_max) on a large random init.
    A full anneal (start_sigma None/0) keeps the dual model in
    distribution, so it is not flagged.  Short-range models are never
    flagged.
    """
    inner = model
    for _ in range(4):  # walk .model wrapper chain (adapter / sharded)
        ct = getattr(inner, "conv_type", None)
        if ct is not None:
            break
        nxt = getattr(inner, "model", None)
        if nxt is None or nxt is inner:
            break
        inner = nxt
    ct = getattr(inner, "conv_type", None)
    # Only the low-σ-start relaxation path is risky; a full anneal is fine.
    uses_relax = start_sigma is not None and float(start_sigma) > 0
    if ct == "dual_cutoff" and uses_relax and n_atoms > _DUAL_CUTOFF_SAFE_ATOMS:
        return (
            f"dual-cutoff (long-range) model + start_sigma relaxation with "
            f"{n_atoms} atoms: at low σ the model's long-range branch "
            f"diverges on the random interface init and the structure "
            f"collapses (overlapping atoms, bad coordination).  This is the "
            f"start_sigma path only — the same model anneals fine from "
            f"σ_max (bulk).  Fixes: use a short-range (graphite / uvu) "
            f"model, which is stable with start_sigma and keeps the phases "
            f"apart; or keep the cell small (< ~{_DUAL_CUTOFF_SAFE_ATOMS} "
            f"atoms); or run a full anneal (start_sigma: null) if some "
            f"interfacial mixing is acceptable.")
    return None


def _axis_index(axis):
    """Map an 'x'/'y'/'z' or 0/1/2 stacking-axis spec to an int in 0..2."""
    if axis not in _AXIS:
        raise ValueError(f"stack_axis must be one of x/y/z or 0/1/2, got {axis!r}")
    return _AXIS[axis]


def _norm_cross_section(cs):
    """Normalise a cross-section spec to a 2x2 in-plane cell matrix or ``None``.

    The rows of the returned matrix are the two in-plane lattice vectors
    (components along the two non-stacking axes), so the cross-section
    can be any parallelogram — not just a rectangle.  Accepts:

    * None            → auto (near-square) or taken from a substrate
    * scalar          → square section, ``diag(L, L)``
    * ``[La, Lb]``    → rectangular section, ``diag(La, Lb)``
    * ``[La, Lb, γ]`` → oblique section: ``a = (La, 0)``,
                        ``b = (Lb·cos γ, Lb·sin γ)`` with γ in degrees
                        (e.g. ``γ = 120`` for a hexagonal surface cell)
    * ``[[ax, ay], [bx, by]]`` → explicit in-plane lattice vectors

    The two vectors must span a non-degenerate cell (|det| > 0).
    """
    if cs is None:
        return None
    if isinstance(cs, (int, float)):
        v = float(cs)
        if v <= 0:
            raise ValueError(f"cross_section must be positive, got {cs!r}.")
        return np.diag([v, v])
    try:
        seq = list(cs)
    except TypeError:
        raise ValueError(
            f"cross_section must be a number, [La, Lb], [La, Lb, gamma_deg] "
            f"or [[ax, ay], [bx, by]], got {cs!r}.")
    # Explicit 2x2 matrix — two rows of two components each.
    if len(seq) == 2 and all(hasattr(r, "__len__") and not isinstance(r, str)
                             for r in seq):
        M = np.array(seq, dtype=float)
        if M.shape != (2, 2):
            raise ValueError(
                f"cross_section matrix must be 2x2 [[ax, ay], [bx, by]], "
                f"got {cs!r}.")
    elif len(seq) == 2:
        La, Lb = float(seq[0]), float(seq[1])
        if La <= 0 or Lb <= 0:
            raise ValueError(
                f"cross_section lengths must be positive, got {cs!r}.")
        M = np.diag([La, Lb])
    elif len(seq) == 3:
        La, Lb, gam = float(seq[0]), float(seq[1]), float(seq[2])
        if La <= 0 or Lb <= 0:
            raise ValueError(
                f"cross_section lengths must be positive, got {cs!r}.")
        if not (0.0 < gam < 180.0):
            raise ValueError(
                f"cross_section angle must be in (0, 180) degrees, got {gam}.")
        g = np.deg2rad(gam)
        M = np.array([[La, 0.0], [Lb * np.cos(g), Lb * np.sin(g)]])
    else:
        raise ValueError(
            f"cross_section must be a number, [La, Lb], [La, Lb, gamma_deg] "
            f"or [[ax, ay], [bx, by]], got {cs!r}.")
    if abs(np.linalg.det(M)) < 1e-8:
        raise ValueError(
            f"cross_section vectors are degenerate (zero area): {cs!r}.")
    return M


def _cross_geom(C2):
    """(|a|, |b|, γ_deg, area) of a 2x2 in-plane cell matrix (rows = vectors)."""
    La = float(np.linalg.norm(C2[0]))
    Lb = float(np.linalg.norm(C2[1]))
    cosg = float(np.dot(C2[0], C2[1]) / (La * Lb))
    gamma = float(np.degrees(np.arccos(np.clip(cosg, -1.0, 1.0))))
    return La, Lb, gamma, float(abs(np.linalg.det(C2)))


def _cross_str(C2):
    """Human-readable '|a|=… |b|=… γ=…°' summary of an in-plane cell."""
    La, Lb, gamma, _ = _cross_geom(C2)
    return f"|a|={La:.3f} |b|={Lb:.3f} γ={gamma:.2f}°"


def _region_mass(elements, counts):
    """Total mass (amu) of a composition given as parallel element/count lists."""
    bad = [e for e in elements if e not in ASE_Z]
    if bad:
        raise ValueError(f"Unknown element symbol(s) {bad} — not in ASE's table.")
    return sum(atomic_masses[ASE_Z[e]] * c for e, c in zip(elements, counts))


def _region_volume(elements, counts, density):
    """Volume (Å³) a composition occupies at the requested mass density (g/cm³)."""
    if density is None or density <= 0:
        raise ValueError(
            f"region density must be a positive number, got {density!r}. "
            "Provide `density` for every non-substrate region.")
    return (_region_mass(elements, counts) * _AMU_TO_GRAM / density) * 1e24


def _place_slab_random(elements, counts, cell2d, z0, z1, min_dist=1.2,
                       seed=None):
    """Random atom placement inside a slab, periodic in-plane, bounded normal.

    The slab spans the full in-plane cell ``cell2d`` — a 2x2 matrix whose
    rows are the in-plane lattice vectors, so the cross-section can be
    any parallelogram (rectangular, hexagonal, oblique) — periodic with
    minimum-image distance, and the normal interval ``[z0, z1)``
    (non-periodic — the inter-slab gap in the assembled cell keeps atoms
    in different slabs apart across the boundary).  Returns
    ``(symbols, positions)`` with positions in the *local* frame
    (in-plane axes 0,1; normal 2).

    Rejection sampling against a spatial hash grid: each candidate is
    checked only against the 27 neighbouring bins, so placement is O(n)
    rather than the O(n²) pure-Python scan that dominated wall time for
    large slabs (9k atoms/layer took minutes; the grid places 36k atoms
    in seconds).  In-plane bins are laid out along the *fractional* axes;
    their perpendicular widths (area/|b| per a-bin, area/|a| per b-bin)
    are kept >= min_dist, so the 3x3x3 block still covers a min_dist
    sphere for any cell shape.  Minimum image uses fractional rounding —
    exact for contact distances << the cell's perpendicular widths.  For
    a diagonal ``cell2d`` this reproduces the legacy rectangular path
    draw-for-draw (same RNG stream, same bins), so seeds are stable.
    """
    syms = [s for s, n in zip(elements, counts) for _ in range(int(n))]
    nt = len(syms)
    md = float(min_dist)
    md2 = md * md
    rng = np.random.default_rng(seed)
    dz = z1 - z0
    if dz <= 0:
        raise ValueError(f"slab thickness must be positive (z0={z0}, z1={z1}).")

    C = np.asarray(cell2d, dtype=float).reshape(2, 2)
    la = np.linalg.norm(C[0])
    lb = np.linalg.norm(C[1])
    area = abs(np.linalg.det(C))
    if area < 1e-8:
        raise ValueError("in-plane cell is degenerate (zero area).")
    # Perpendicular widths of the fractional bins: stepping one bin along
    # fractional axis a moves the point by C[0]/nx, whose component
    # normal to b is (area/|b|)/nx — that is the covering length that
    # must stay >= min_dist (and symmetrically for b).
    w_a = area / lb
    w_b = area / la

    # Bin counts: perpendicular edge >= min_dist so a sphere of radius
    # min_dist around any point is covered by the 3x3x3 block around it.
    nx = max(1, int(w_a / md))
    ny = max(1, int(w_b / md))
    nz = max(1, int(dz / md))
    grid = {}                     # (bx, by, bz) -> list of atom indices

    fr = np.empty((nt, 2))        # in-plane fractional coords
    pz = np.empty(nt)             # normal (cartesian) coords
    for k in range(nt):
        placed = False
        for _ in range(20_000):
            fa, fb = rng.random(), rng.random()
            cz = z0 + rng.random() * dz
            bx = min(int(fa * nx), nx - 1)
            by = min(int(fb * ny), ny - 1)
            bz = min(int((cz - z0) / dz * nz), nz - 1)
            ok = True
            for ix in (bx - 1, bx, bx + 1):
                for iy in (by - 1, by, by + 1):
                    for iz in (bz - 1, bz, bz + 1):
                        if iz < 0 or iz >= nz:      # z is non-periodic
                            continue
                        cell_atoms = grid.get((ix % nx, iy % ny, iz))
                        if not cell_atoms:
                            continue
                        # Minimum image in the periodic in-plane axes only,
                        # via fractional rounding (exact for d < w/2).
                        dfr = np.array([fa, fb]) - fr[cell_atoms]
                        dfr -= np.round(dfr)
                        dxy = dfr @ C
                        dzv = cz - pz[cell_atoms]
                        if (dxy[:, 0] ** 2 + dxy[:, 1] ** 2
                                + dzv * dzv < md2).any():
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                fr[k] = (fa, fb)
                pz[k] = cz
                grid.setdefault((bx, by, bz), []).append(k)
                placed = True
                break
        if not placed:
            raise RuntimeError(
                f"Cannot place atom {k} ({syms[k]}) in slab — too dense. "
                f"Lower `min_dist`, lower the region density, or thicken "
                f"the slab.")
    pos = np.empty((nt, 3))
    pos[:, :2] = fr @ C
    pos[:, 2] = pz
    return syms, pos


def _resolve_region(spec):
    """Normalise a region spec into a common dict.

    A region is either a *substrate* (fixed atoms read from a POSCAR /
    ASE-readable file, or an ``ase.Atoms`` passed directly) or a *random
    amorphous* slab (elements + counts + density).  Returns a dict with:
      kind          : 'substrate' | 'random'
      symbols       : list[str]                     (substrate only)
      positions     : (n,3) local-frame array       (substrate only)
      cell          : (3,3) substrate cell           (substrate only)
      thickness     : Lz or None                    (substrate defines it)
      elements/counts/density                       (random only)
    """
    if spec is None:
        raise ValueError("interface region spec is missing.")

    atoms = spec.get("atoms")
    path = spec.get("poscar") or spec.get("path")
    if atoms is not None or path is not None:
        if atoms is None:
            import ase.io
            atoms = ase.io.read(path, format="vasp")
        # Any cell shape is accepted here; geometric validity against the
        # stacking axis (in-plane vectors ⊥ axis, stacking vector along
        # the axis) is checked in `build_multilayer_structure`, which
        # knows the axis.
        return {
            "kind": "substrate",
            "symbols": list(atoms.get_chemical_symbols()),
            "positions": np.array(atoms.positions, dtype=float),
            "cell": np.array(atoms.cell, dtype=float),
            "coord_cutoff": dict(spec.get("coord_cutoff", {}) or {}),
            # Optional explicit region id (else the layer index is used).
            # Lets repeated phases share a label, e.g. SiO2/GeO2/SiO2 with
            # region=[0,1,0] for a region-aware model with n_regions=2.
            "region_id": spec.get("region"),
        }

    elements = spec.get("elements")
    counts = spec.get("counts")
    if not elements or not counts or len(elements) != len(counts):
        raise ValueError(
            "random region needs parallel `elements` and `counts` lists "
            f"of equal length; got elements={elements}, counts={counts}.")
    return {
        "kind": "random",
        "elements": list(elements),
        "counts": [int(c) for c in counts],
        "density": spec.get("density"),
        "coord_cutoff": dict(spec.get("coord_cutoff", {}) or {}),
        "region_id": spec.get("region"),
    }


def _norm_fixed_layers(fixed_layers, n):
    """Normalise a fixed-layers spec into a set of layer indices in 0..n-1.

    Accepts: None / "none" / [] (nothing pinned), "all", a list of int
    indices, or a length-n list of booleans.
    """
    if fixed_layers is None:
        return set()
    if isinstance(fixed_layers, str):
        s = fixed_layers.strip().lower()
        if s in ("", "none"):
            return set()
        if s == "all":
            return set(range(n))
        raise ValueError(
            f"fixed_layers string must be 'none' or 'all', got {fixed_layers!r}.")
    seq = list(fixed_layers)
    if len(seq) == n and all(isinstance(b, (bool, np.bool_)) for b in seq):
        return {i for i, b in enumerate(seq) if b}
    out = set()
    for v in seq:
        i = int(v)
        if not (0 <= i < n):
            raise ValueError(
                f"fixed_layers index {i} out of range for {n} layers (0..{n-1}).")
        out.add(i)
    return out


def build_multilayer_structure(regions, stack_axis="z", gap=0.8,
                               fixed_layers=None, cross_section=None,
                               min_dist=1.2, seed=None):
    """Assemble N regions into one periodic multilayer cell.

    Layers are stacked in list order along `stack_axis`, each separated
    by `gap` (including a gap after the last layer, which wraps to the
    first across the periodic boundary — so an N-layer stack has N
    interfaces).  Example: ``[SiO2, GeO2, TiO2, SiO2]`` gives a
    SiO2/GeO2, GeO2/TiO2, TiO2/SiO2 and (periodic) SiO2/SiO2 interface.

    Parameters
    ----------
    regions : list of dict
        Region specs (see `_resolve_region`), low → high along the axis.
        Each is a random amorphous slab (``elements``/``counts``/
        ``density``) or a substrate (``poscar``/``atoms``); an optional
        ``region`` key sets its region id for a region-aware model
        (default = layer index).
    stack_axis : 'x'|'y'|'z' or 0|1|2
    gap : float
        Separation (Å) at each interface; clamped up to `min_dist`.
    fixed_layers : None | 'none' | 'all' | list[int] | list[bool]
        Which layers are pinned substrates (fixed_mask + RePaint).
    cross_section : float | (La, Lb) | (La, Lb, gamma_deg) | 2x2 rows | None
        In-plane cross-section (scalar = square, 2 values = rectangle,
        3 values = oblique parallelogram with angle γ in degrees — e.g.
        ``[La, Lb, 120]`` for a hexagonal surface cell — or an explicit
        2x2 matrix of in-plane lattice vectors).  A substrate defines
        it; else an explicit value; else auto (~cubic from total volume).
    min_dist : float
    seed : int or None

    Returns
    -------
    dict: atoms, region_labels (N,), fixed_mask (N,3) or None,
    coord_cutoff, info (with a per-layer ``layers`` list).
    """
    if not regions or len(regions) < 1:
        raise ValueError("build_multilayer_structure needs >= 1 region.")
    ax = _axis_index(stack_axis)
    rs = [_resolve_region(r) for r in regions]
    nL = len(rs)
    gap = max(float(gap), float(min_dist))
    cross_section = _norm_cross_section(cross_section)
    fixed = _norm_fixed_layers(fixed_layers, nL)
    plane = [i for i in range(3) if i != ax]

    # ── 1. Shared in-plane cross-section (2x2 in-plane cell matrix) ──
    # Every substrate must agree on it: substrate atoms keep their raw
    # in-plane coordinates, so a mismatched second substrate would put
    # atoms outside the assembled cell (or at the wrong density) without
    # any error.  Take the first substrate's a,b and verify the rest.
    # The in-plane cell may be any parallelogram (e.g. hexagonal for a
    # quartz (0001) surface) — the assembled cell is then triclinic,
    # which the neighbour-list / generation stack fully supports.
    def _sub_inplane(r, i):
        cell = r["cell"]
        # In-plane lattice vectors must lie in the plane ⊥ stack axis …
        if not np.allclose(cell[plane][:, ax], 0.0, atol=1e-6):
            raise ValueError(
                f"substrate layer {i}: its in-plane lattice vectors have a "
                f"component along the stacking axis "
                f"({'xyz'[ax]}). Re-cut the slab so the two surface vectors "
                f"are perpendicular to the stacking axis.")
        # … and the stacking vector must be along the axis (no tilt): the
        # assembled cell is a straight prism (random slabs + gaps are
        # built along the axis), so a tilted stacking vector cannot be
        # represented.  Re-cut the slab with c ⊥ the surface plane.
        if not np.allclose(cell[ax][plane], 0.0, atol=1e-6):
            raise ValueError(
                f"substrate layer {i}: its stacking lattice vector is tilted "
                f"(has in-plane components {cell[ax][plane]}). Re-cut the "
                f"slab with the stacking vector perpendicular to the surface "
                f"plane (an orthogonal c axis).")
        if cell[ax, ax] <= 0:
            raise ValueError(
                f"substrate layer {i}: non-positive height along the "
                f"stacking axis ({cell[ax, ax]}).")
        return cell[np.ix_(plane, plane)].astype(float)

    sub_cross = None
    for i, r in enumerate(rs):
        if r["kind"] == "substrate":
            this = _sub_inplane(r, i)
            if sub_cross is None:
                sub_cross = this
            elif not np.allclose(this, sub_cross, atol=1e-3):
                raise ValueError(
                    f"substrate layer {i} has in-plane cross-section "
                    f"[{_cross_str(this)}] but an earlier substrate set "
                    f"[{_cross_str(sub_cross)}]; all substrate layers must "
                    "share the same a,b (re-cut or supercell one of them to "
                    "match).")
    if sub_cross is not None:
        C2 = sub_cross
        if cross_section is not None and not np.allclose(
                cross_section, C2, atol=1e-3):
            raise ValueError(
                f"cross_section [{_cross_str(cross_section)}] conflicts with "
                f"the substrate's cross-section [{_cross_str(C2)}].  Omit "
                f"cross_section, or make them match.")
    elif cross_section is not None:
        C2 = cross_section
    else:
        # No substrate, no explicit section: square section so the
        # assembled cell is roughly cube-shaped over all layers.
        v_tot = sum(_region_volume(r["elements"], r["counts"], r["density"])
                    for r in rs if r["kind"] == "random")
        if v_tot <= 0:
            raise ValueError(
                "Cannot auto-size the cross-section: no random layer with a "
                "density and no substrate to take a,b from.  Provide "
                "`cross_section`.")
        L = v_tot ** (1.0 / 3.0)
        C2 = np.diag([L, L])
    La, Lb, gamma, area = _cross_geom(C2)

    def _thickness(r, i):
        """Atom-placement thickness of layer ``i`` (Å).

        Each layer's ALLOCATED region in the periodic cell is its slab
        plus one trailing ``gap`` (N layers → N gaps), and that gap is
        empty volume.  If a random layer's atoms were spread over the
        full V/area, the layer's realized density in the assembled cell
        would be ρ·Lz/(Lz+gap) — systematically low (≈2% for an 85 Å
        slab at gap 1.4, ~14% for a 10 Å slab).  So random layers are
        placed in ``V/area − gap``: slab + gap = V/area exactly, and the
        realized (allocated-region) density equals the requested one.
        Substrates keep their own cell height — a fixed structure cannot
        be rescaled — so their trailing gap still dilutes; the per-layer
        ``density_realized`` in `info` makes that visible.
        """
        if r["kind"] == "substrate":
            return float(r["cell"][ax, ax])
        lz = _region_volume(r["elements"], r["counts"], r["density"]) / area - gap
        if lz < max(float(min_dist), 1.0):
            raise ValueError(
                f"layer {i}: slab is too thin ({lz + gap:.2f} Å allocated) "
                f"to absorb the {gap:.2f} Å interface gap at density "
                f"{r['density']} g/cm³. Increase the atom counts, reduce "
                f"cross_section, or reduce gap.")
        return lz

    # ── 2. Place each layer, low → high, with a gap after each ──
    def _local_positions(r, z_lo, z_hi, rng_seed):
        if r["kind"] == "substrate":
            p = r["positions"].copy()
            local = np.empty_like(p)
            local[:, 0] = p[:, plane[0]]
            local[:, 1] = p[:, plane[1]]
            local[:, 2] = p[:, ax] - p[:, ax].min() + z_lo
            return list(r["symbols"]), local
        return _place_slab_random(r["elements"], r["counts"], C2,
                                  z_lo, z_hi, min_dist=min_dist, seed=rng_seed)

    def _to_global(local):
        g = np.empty_like(local)
        g[:, plane[0]] = local[:, 0]
        g[:, plane[1]] = local[:, 1]
        g[:, ax] = local[:, 2]
        return g

    all_pos, all_sym, all_region = [], [], []
    fixed_flags = []
    layers_info = []
    coord_cutoff = {}
    z_cursor = 0.0
    thicknesses = []
    for i, r in enumerate(rs):
        Lz = _thickness(r, i)
        thicknesses.append(Lz)
        z_lo, z_hi = z_cursor, z_cursor + Lz
        rng_seed = None if seed is None else int(seed) + i * 104729
        syms, loc = _local_positions(r, z_lo, z_hi, rng_seed)
        all_pos.append(_to_global(loc))
        all_sym += syms
        rid = r["region_id"] if r.get("region_id") is not None else i
        all_region += [int(rid)] * len(syms)
        is_fixed = i in fixed
        fixed_flags += [is_fixed] * len(syms)
        coord_cutoff.update(r.get("coord_cutoff", {}) or {})
        # Realized density of the layer's allocated region (slab + its
        # trailing gap) — equals the requested density for random layers
        # after the gap compensation in `_thickness`; substrates show
        # their gap-diluted value.
        m_amu = (_region_mass(r["elements"], r["counts"])
                 if r["kind"] == "random" else
                 sum(atomic_masses[ASE_Z[s]] for s in r["symbols"]))
        rho_real = m_amu * _AMU_TO_GRAM / (area * (Lz + gap)) * 1e24
        layers_info.append({
            "index": i, "kind": r["kind"], "n": len(syms), "region": int(rid),
            "thickness": round(float(Lz), 3),
            "z_range": (round(float(z_lo), 3), round(float(z_hi), 3)),
            "fixed": is_fixed,
            "density_requested": (r.get("density") if r["kind"] == "random"
                                  else None),
            "density_realized": round(float(rho_real), 4),
            "elements": (list(r["elements"]) if r["kind"] == "random"
                         else sorted(set(r["symbols"]))),
        })
        z_cursor = z_hi + gap

    total_z = z_cursor  # sum(thicknesses) + nL*gap
    pos = np.vstack(all_pos)
    n_total = len(all_sym)

    # Assembled cell: a straight prism — the in-plane 2x2 block carries
    # the (possibly oblique) cross-section, the stacking vector is purely
    # along the axis.  Diagonal in-plane block → orthorhombic, as before.
    cell3 = np.zeros((3, 3))
    cell3[np.ix_(plane, plane)] = C2
    cell3[ax, ax] = total_z
    atoms = Atoms(symbols=all_sym, positions=pos, cell=cell3, pbc=True)

    region_labels = np.asarray(all_region, dtype=np.int64)
    fixed_mask = None
    if any(fixed_flags):
        fixed_mask = np.zeros((n_total, 3), dtype=bool)
        fixed_mask[np.asarray(fixed_flags, dtype=bool)] = True

    # Whole-cell mass density (g/cm³) — the mass-weighted average of the
    # layers; matches the requested densities when no substrate dilutes.
    cell_mass = sum(atomic_masses[ASE_Z[s]] for s in all_sym)
    cell_rho = cell_mass * _AMU_TO_GRAM / abs(float(np.linalg.det(cell3))) * 1e24

    info = {
        "n_layers": nL,
        "n_total": n_total,
        "layers": layers_info,
        # (|a|, |b|) lengths — for an orthorhombic section these are the
        # legacy (La, Lb); `cross_angle` (γ, degrees; 90 = rectangular)
        # and `cross_matrix` carry the full in-plane shape.
        "cross_section": (round(float(La), 3), round(float(Lb), 3)),
        "cross_angle": round(float(gamma), 3),
        "cross_matrix": tuple(tuple(round(float(v), 4) for v in row)
                              for row in C2),
        "thicknesses": [round(float(t), 3) for t in thicknesses],
        "gap": round(float(gap), 3),
        # Row lengths of the assembled cell (== diagonal for orthorhombic
        # cells, the legacy value); `cell_matrix` is the full 3x3.
        "cell": tuple(round(float(np.linalg.norm(cell3[i])), 3)
                      for i in range(3)),
        "cell_matrix": tuple(tuple(round(float(v), 4) for v in row)
                             for row in cell3),
        "density_cell": round(float(cell_rho), 4),
        "stack_axis": {0: "x", 1: "y", 2: "z"}[ax],
        "fixed_layers": sorted(fixed),
    }
    return {"atoms": atoms, "region_labels": region_labels,
            "fixed_mask": fixed_mask, "coord_cutoff": coord_cutoff,
            "info": info}


def build_interface_structure(region_a, region_b, stack_axis="z", gap=0.8,
                              fix_region="a", cross_section=None,
                              min_dist=1.2, seed=None):
    """Two-region interface — thin wrapper over `build_multilayer_structure`.

    Region A is the low side, region B the high side.  `fix_region`
    ('a'|'b'|'none') selects which side is pinned.  The returned `info`
    carries the general multilayer keys plus the legacy 2-region keys
    (``n_a``/``n_b``/``kind_a``/``kind_b``/``thickness_a``/
    ``thickness_b``/``fixed``) for backward compatibility.  See
    `build_multilayer_structure` for the full parameter semantics.
    """
    fix = (fix_region or "none").strip().lower()
    if fix not in ("a", "b", "none"):
        raise ValueError(f"fix_region must be 'a', 'b', or 'none', got {fix_region!r}")
    fixed_layers = {"a": [0], "b": [1], "none": []}[fix]
    built = build_multilayer_structure(
        [region_a, region_b], stack_axis=stack_axis, gap=gap,
        fixed_layers=fixed_layers, cross_section=cross_section,
        min_dist=min_dist, seed=seed)
    L = built["info"]["layers"]
    built["info"].update({
        "n_a": L[0]["n"], "n_b": L[1]["n"],
        "kind_a": L[0]["kind"], "kind_b": L[1]["kind"],
        "thickness_a": L[0]["thickness"], "thickness_b": L[1]["thickness"],
        "fixed": fix,
    })
    return built


def generate_interface(model, built, target_energy, device,
                       noise, refine, snap_every, plot_every,
                       gnn_cut, vskin, *, repaint=True, use_region=None,
                       all_model_elements=None, start_sigma=None,
                       max_disp_denoise=0.0, max_disp_refine=0.0,
                       min_dist_constraint=0.0,
                       sigma_schedule="edm", sigma_rho=7.0,
                       v2_cond=None, com_detrend="xy",
                       stop_fn=None, on_snap=None, on_plot=None):
    """Run interface generation on an already-built structure.

    `built` is the dict returned by `build_interface_structure`.  Fixed
    atoms (per `built['fixed_mask']`) are held via RePaint; the film and
    boundary atoms are denoised.  Region labels are attached to the graph
    so a region-aware model conditions on them (bulk models ignore them).

    `use_region` defaults to "auto": attach region labels iff the model
    was built region-aware (``n_regions > 0``).

    `start_sigma` caps the starting noise level (forwarded as
    `generate(poscar_start_sigma=...)`).  This is the key control for a
    two-*amorphous*-phase interface with a bulk model: the builder places
    each phase in its own slab, but running the full σ_max schedule from
    scratch scrambles that layering into a homogeneous mixed glass,
    because a bulk model has no signal to keep the phases apart.  Capping
    σ (e.g. 0.20–0.35) *relaxes* the stacked slab — atoms order locally
    into their coordination polyhedra — without letting them diffuse
    across the interface, so the compositional gradient is preserved.
    None → full schedule (correct when a substrate is pinned, or for a
    single-phase relaxation).
    """
    # Imported here (not at module top) so `build_interface_structure`
    # stays importable without torch — the builder is pure NumPy/ASE.
    from dit2.generation.core import generate, mk_data, _get_training_z
    from dit2.generation.multigpu import unwrap_model

    atoms = built["atoms"]
    fixed_mask = built["fixed_mask"]
    region_labels = built["region_labels"]

    # Walk wrapper chains (ShardedModel → GraphiteModelAdapter → net) so
    # region-awareness is detected regardless of how the model is wrapped.
    inner = unwrap_model(model)
    n_regions = int(getattr(inner, "n_regions", 0) or 0)
    if use_region is None:
        use_region = n_regions > 0
    region_arg = region_labels if (use_region and n_regions > 0) else None
    # A region id >= n_regions would index past the embedding table and
    # crash mid-generation (CUDA device assert).  Disable region
    # conditioning instead — same graceful fallback as a bulk model.
    if region_arg is not None and int(np.max(region_arg)) >= n_regions:
        import warnings
        warnings.warn(
            f"interface: region ids reach {int(np.max(region_arg))} but the "
            f"model has n_regions={n_regions}; disabling region conditioning "
            "(labels would index past the embedding table).",
            RuntimeWarning, stacklevel=2)
        region_arg = None

    # Conditioning v2 (method / T_eff / ΔE / RDF + cond_weights): same
    # kwargs contract as the bulk path (`_build_v2_conditioning` output).
    # None/empty → legacy energy-only behaviour, bit-identical.
    data = mk_data(atoms, all_model_elements or [], target_energy, device,
                   fixed_mask=fixed_mask,
                   training_z=_get_training_z(model),
                   region=region_arg, **(v2_cond or {}))

    return generate(model, data, atoms, noise, refine, snap_every,
                    plot_every, gnn_cut, vskin, device,
                    repaint=repaint, poscar_start_sigma=start_sigma,
                    min_dist_constraint=min_dist_constraint,
                    max_disp_denoise=max_disp_denoise,
                    max_disp_refine=max_disp_refine,
                    sigma_schedule=sigma_schedule, sigma_rho=sigma_rho,
                    com_detrend=com_detrend,
                    stop_fn=stop_fn, on_snap=on_snap, on_plot=on_plot)
