"""Small physics-related helpers shared across the package.

Kept separate from heavier modules so ``dit2.utils.logging`` and other
low-level utilities can pull these helpers in without dragging
``torch`` / ``e3nn`` into the import graph.
"""

_AMU_PER_A3_TO_G_PER_CC = 1.66054

# Multiplicative factor: amu × this = grams.  Used in structure builders
# that convert (target ρ in g/cm³, sum-of-atomic-masses in amu) → cell
# volume in Å³.  Centralizing this here lets every call site stay in
# sync if the canonical value is ever updated to 1.66053906660e-24
# (CODATA 2018).
_AMU_TO_GRAM = 1.66054e-24


def _mass_density(atoms) -> float:
    """Convert an ASE ``Atoms`` object to mass density in g/cm³.

    Returns ``float('nan')`` on any error (e.g. zero-volume cell) so
    callers can filter those frames out without raising.
    """
    try:
        return (atoms.get_masses().sum() / atoms.get_volume()
                * _AMU_PER_A3_TO_G_PER_CC)
    except Exception:
        return float('nan')
