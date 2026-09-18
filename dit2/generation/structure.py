from io import StringIO

import ase.io
import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers as ASE_Z
from ase.io import read

# M12: single source of truth for the amu → gram conversion factor.
from dit2.utils.physics import _AMU_TO_GRAM


def read_poscar(path, allowed=None):
    """Read POSCAR with selective-dynamics support.

    Returns
    -------
    atoms : ase.Atoms | None
    info  : dict | None   – includes 'fixed_mask' (N,3) bool array
    err   : str | None
    """
    try: atoms = ase.io.read(path, format="vasp")
    except Exception as e: return None, None, str(e)
    syms = atoms.get_chemical_symbols()
    uq = []; cts = {}
    for s in syms:
        if s not in cts: uq.append(s); cts[s] = 0
        cts[s] += 1
    if allowed:
        bad = [e for e in uq if e not in allowed]
        if bad: return None, None, f"Elements {bad} not in {allowed}"
    atoms.pbc = True
    mass = sum(atomic_masses[ASE_Z[s]] for s in syms)
    vol = atoms.get_volume()
    rho = (mass * _AMU_TO_GRAM) / (vol * 1e-24)

    # ---- selective dynamics ----
    fixed_mask = np.zeros((len(atoms), 3), dtype=bool)   # True = FIXED
    sd_source = None

    if 'selective_dynamics' in atoms.arrays:
        # ASE convention: True = free, False = fixed
        sd_flags = np.asarray(atoms.arrays['selective_dynamics'], dtype=bool)
        fixed_mask = ~sd_flags
        sd_source = "selective_dynamics"
    else:
        from ase.constraints import FixAtoms, FixCartesian
        # FixScaled is what ASE ≥ 3.23 produces when reading a Direct
        # POSCAR with per-component selective dynamics; older files gave
        # FixCartesian.  Both carry a (3,) mask (True = fixed) over one or
        # more atom indices, so handle them together with broadcasting.
        try:
            from ase.constraints import FixScaled
        except ImportError:
            FixScaled = ()
        for c in atoms.constraints:
            if isinstance(c, FixAtoms):
                fixed_mask[c.index, :] = True
                sd_source = "FixAtoms"
            elif isinstance(c, (FixCartesian,) + ((FixScaled,) if FixScaled else ())):
                idx = np.atleast_1d(np.asarray(c.index))
                mask = np.asarray(c.mask, dtype=bool).reshape(-1)  # (3,)
                fixed_mask[idx] = mask  # broadcast (k,3) <- (3,)
                sd_source = type(c).__name__

    n_fixed_full = int(np.all(fixed_mask, axis=1).sum())
    n_fixed_part = int(np.any(fixed_mask, axis=1).sum()) - n_fixed_full

    info = {"types": uq, "counts": [cts[e] for e in uq], "density": round(rho, 4),
            "n_atoms": len(atoms), "cell": tuple(round(x, 3) for x in atoms.cell.lengths()),
            "volume": round(vol, 2), "formula": " ".join(f"{e}{cts[e]}" for e in uq),
            "fixed_mask": fixed_mask,
            "n_fixed_full": n_fixed_full,
            "n_fixed_partial": n_fixed_part,
            "sd_source": sd_source}
    return atoms, info, None

###############################################################################
# POSCAR WRITER  (FIX: preserve selective dynamics in output)
###############################################################################
def write_poscar_with_sd(atoms, fixed_mask, stream_or_path):
    """Write VASP POSCAR preserving selective dynamics.

    atoms      : ase.Atoms
    fixed_mask : np.ndarray (N,3) bool, True=fixed.  None → no SD.
    """
    if fixed_mask is not None and np.any(fixed_mask):
        # Modern ASE (≥ 3.23) derives the selective-dynamics F/T flags from
        # `atoms.constraints`, ignoring the legacy
        # `arrays['selective_dynamics']` on write; and in ASE 3.28 the vasp
        # writer honours only FixAtoms, dropping FixCartesian entirely.  So
        # a partial (per-component) pin would be silently lost, and a
        # downstream VASP/MD relaxation would move atoms meant to stay
        # fixed (e.g. an interface substrate).  Rather than depend on that
        # conversion, write the base POSCAR without constraints and inject
        # the "Selective dynamics" block + per-atom F/T columns ourselves.
        # ASE writes atoms in input order (sort=False default), so
        # `fixed_mask[i]` lines up with the i-th coordinate row.
        fm = np.asarray(fixed_mask, dtype=bool)
        buf = StringIO()
        ase.io.write(buf, atoms, format="vasp", vasp5=True, direct=True)
        lines = buf.getvalue().splitlines()
        # Locate the coordinate-mode line ("Direct"/"Cartesian"); the N
        # coordinate rows follow it.
        ci = next(i for i, ln in enumerate(lines)
                  if ln.strip()[:1] in ("D", "d", "C", "c"))
        out = lines[:ci] + ["Selective dynamics"] + [lines[ci]]
        for k in range(fm.shape[0]):
            coord = lines[ci + 1 + k]
            flags = "".join(f"  {'F' if fm[k, j] else 'T'}" for j in range(3))
            out.append(coord + flags)
        out.extend(lines[ci + 1 + fm.shape[0]:])
        text = "\n".join(out) + "\n"
        if hasattr(stream_or_path, "write"):
            stream_or_path.write(text)
        else:
            with open(stream_or_path, "w") as f:
                f.write(text)
    else:
        # Write in direct (fractional) coordinates consistently so the
        # output is round-trip-stable regardless of whether selective
        # dynamics is present.  Earlier revisions used Cartesian here,
        # which silently changed the coordinate basis between SD and
        # non-SD writes.
        ase.io.write(stream_or_path, atoms, format="vasp",
                     vasp5=True, direct=True)

###############################################################################
# STRUCTURE CREATION
###############################################################################
def _normalize_cell_vectors(cv):
    """Normalize cell vectors to unit volume (det=1) so they are a pure shape template.
    Accepts any array-like (3,3). Raises ValueError on bad input."""
    cv = np.array(cv, dtype=float)
    if cv.shape != (3, 3):
        raise ValueError(f"cell_vectors must be 3×3, got shape {cv.shape}")
    V = abs(np.linalg.det(cv))
    if V < 1e-12:
        raise ValueError(f"Cell vectors are degenerate (det≈0): {cv.tolist()}")
    return cv / (V ** (1.0 / 3.0))   # scale so det = 1

def create_struct(ty, ct, rho, md=1.2, cell_vectors=None, seed=None):
    """Create a random structure with the given composition and density.

    Parameters
    ----------
    cell_vectors : array-like (3,3), optional
        Lattice vectors [[ax,ay,az],[bx,by,bz],[cx,cy,cz]] defining the
        cell SHAPE. Automatically normalized to unit volume, then scaled
        so the cell volume matches the target density. None → cubic cell.
    seed : int, optional
        RNG seed for reproducible initial placement.  None → fresh
        non-deterministic RNG (legacy behaviour).
    """
    syms = [s for s, n in zip(ty, ct) for _ in range(n)]; nt = len(syms)
    # Validate element symbols against ASE early so an unknown symbol
    # raises a clear message instead of a cryptic KeyError mid-loop.
    bad = [s for s in ty if s not in ASE_Z]
    if bad:
        raise ValueError(
            f"Unknown element symbol(s) {bad} — not found in ASE's "
            f"atomic_numbers table. Check your composition.")
    mass = sum(atomic_masses[ASE_Z[s]]*c for s, c in zip(ty, ct))
    V_target = (mass * _AMU_TO_GRAM / rho) * 1e24   # required volume in ų

    if cell_vectors is not None:
        cv = _normalize_cell_vectors(cell_vectors)   # det = 1
        cell = cv * (V_target ** (1.0 / 3.0))       # scale to target volume
    else:
        L = V_target ** (1.0 / 3.0)
        cell = np.diag([L, L, L])

    md2 = md ** 2
    rng = np.random.default_rng(seed)
    frac_pos = np.empty((nt, 3))
    cart_pos = np.empty((nt, 3))

    # For non-orthogonal cells, the fractional-cube min-image
    # (`df -= round(df)`) does not always pick the true Cartesian
    # shortest image — the nearest periodic image of a fractional
    # offset near a corner can lie in a different (±1, ±1, ±1)
    # fractional octant.  Enumerate 27 Cartesian shifts and pick
    # the actual minimum.
    ortho = is_ortho(cell)
    if not ortho:
        _shifts = np.array(
            [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1)
             for k in (-1, 0, 1)], dtype=float)

    for k in range(nt):
        ok = False
        for _ in range(10_000):
            fp = rng.random(3)                  # fractional [0,1)
            cp = fp @ cell                       # Cartesian
            good = True
            for i in range(k):
                df = frac_pos[i] - fp
                df -= np.round(df)               # wrap to [-0.5, 0.5)
                if ortho:
                    dc = df @ cell
                    d2_min = dc[0]*dc[0] + dc[1]*dc[1] + dc[2]*dc[2]
                else:
                    # 27-shift Cartesian enumeration: (27, 3)
                    dc_all = (df + _shifts) @ cell
                    d2_min = float((dc_all * dc_all).sum(axis=1).min())
                if d2_min < md2:
                    good = False; break
            if good:
                frac_pos[k] = fp
                cart_pos[k] = cp
                ok = True; break
        if not ok:
            raise RuntimeError(f"Cannot place atom {k} ({syms[k]})")

    return Atoms(symbols=syms, positions=cart_pos, cell=cell, pbc=True)

def create_struct_gpu(ty, ct, rho, dv, md=1.2, cell_vectors=None, seed=None):
    """GPU-accelerated random structure initialization using batched candidate generation."""
    syms = [s for s, n in zip(ty, ct) for _ in range(n)]
    nt = len(syms)
    bad = [s for s in ty if s not in ASE_Z]
    if bad:
        raise ValueError(
            f"Unknown element symbol(s) {bad} — not found in ASE's "
            f"atomic_numbers table. Check your composition.")
    mass = sum(atomic_masses[ASE_Z[s]]*c for s, c in zip(ty, ct))
    V_target = (mass * _AMU_TO_GRAM / rho) * 1e24   # required volume in A^3

    if cell_vectors is not None:
        cv = _normalize_cell_vectors(cell_vectors)
        cell_np = cv * (V_target ** (1.0 / 3.0))
    else:
        L = V_target ** (1.0 / 3.0)
        cell_np = np.diag([L, L, L])

    # Move logic to target device
    cell = torch.tensor(cell_np, dtype=torch.float32, device=dv)
    md2 = md ** 2

    # Reproducible RNG when `seed` is given; default torch global RNG
    # otherwise (preserves legacy behaviour for callers that don't pass
    # a seed).
    if seed is not None:
        gen = torch.Generator(device=dv).manual_seed(int(seed))
    else:
        gen = None

    frac_pos = torch.empty((nt, 3), dtype=torch.float32, device=dv)
    cart_pos = torch.empty((nt, 3), dtype=torch.float32, device=dv)

    # Pre-build 27 Cartesian shifts for triclinic cells — fractional-cube
    # min-image (`diff -= round(diff)`) does not always pick the true
    # Cartesian shortest image for non-orthogonal lattices.
    ortho = is_ortho(cell)
    if not ortho:
        _shifts = torch.tensor(
            [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1)
             for k in (-1, 0, 1)],
            dtype=torch.float32, device=dv)  # (27, 3)

    for k in range(nt):
        ok = False
        # Generate 100 candidate positions at once to parallelize the distance checks
        batch_size = 100

        for _ in range(100):  # Max 10,000 total attempts
            fp_batch = torch.rand((batch_size, 3), dtype=torch.float32,
                                  device=dv, generator=gen)

            if k == 0:
                frac_pos[0] = fp_batch[0]
                cart_pos[0] = fp_batch[0] @ cell
                ok = True
                break

            # Vectorized minimum image distance check: candidates vs. established atoms
            # fp_batch: (B, 3), frac_pos[:k]: (K, 3) -> diff: (B, K, 3)
            diff = frac_pos[:k].unsqueeze(0) - fp_batch.unsqueeze(1)
            diff -= torch.round(diff)
            if ortho:
                dc = diff @ cell
                d2 = (dc**2).sum(dim=-1)  # (B, K)
            else:
                # diff: (B, K, 3); add shifts: (B, K, 27, 3) → Cartesian
                # → squared distance per shift → min over 27 shifts.
                diff_all = diff.unsqueeze(2) + _shifts.view(1, 1, -1, 3)
                dc_all = diff_all @ cell  # (B, K, 27, 3)
                d2 = (dc_all ** 2).sum(dim=-1).min(dim=-1).values  # (B, K)

            # Find candidates where distance to ALL existing atoms is >= md^2
            valid_mask = (d2 >= md2).all(dim=1)
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]

            if valid_indices.numel() > 0:
                best_idx = valid_indices[0]
                frac_pos[k] = fp_batch[best_idx]
                cart_pos[k] = fp_batch[best_idx] @ cell
                ok = True
                break

        if not ok:
            raise RuntimeError(f"Cannot place atom {k} ({syms[k]}) - packing too dense.")

    return Atoms(symbols=syms, positions=cart_pos.cpu().numpy(), cell=cell_np, pbc=True)

def create_struct_auto(ty, ct, rho, dv, md=1.2, cell_vectors=None, seed=None):
    """Auto-routes structure creation to GPU or CPU based on device."""
    if dv.type in ("cuda", "mps"):
        return create_struct_gpu(ty, ct, rho, dv, md=md,
                                  cell_vectors=cell_vectors, seed=seed)
    else:
        return create_struct(ty, ct, rho, md=md,
                              cell_vectors=cell_vectors, seed=seed)

###############################################################################
# CELL GEOMETRY HELPERS
###############################################################################
def is_ortho(cell):
    """Check if cell matrix is orthogonal (diagonal).

    Works on torch tensors without CPU roundtrip — important because this
    is called once per sample from the dataset transform.
    """
    if isinstance(cell, torch.Tensor):
        # Off-diagonal mask: 1 everywhere except the main diagonal
        c = cell.detach()
        # abs() of off-diagonal elements must all be near zero
        n = c.shape[-1]
        eye = torch.eye(n, dtype=torch.bool, device=c.device)
        off = c.masked_select(~eye)
        return bool(off.abs().max().item() < 1e-6) if off.numel() else True
    c = np.asarray(cell)
    off = c.copy(); np.fill_diagonal(off, 0)
    return np.allclose(off, 0, atol=1e-6)
