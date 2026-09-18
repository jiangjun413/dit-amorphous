"""Shared constants used across CLI, UI, and generation modules.

These were originally defined at module scope in the monolithic
``dit10.py``.  Extracting them here avoids a circular dependency
between ``dit2.cli`` (which is a higher-level orchestration module)
and ``dit.generation.analysis`` (which is a leaf module that needs
the same constants for plotting / legend rendering).
"""

import os
from ase.data import atomic_numbers as ASE_Z
from ase.data.colors import jmol_colors as _ASE_JMOL


# Legacy default element list (kept only for backward compatibility with
# old callers that import this constant).  The model now embeds atoms by
# atomic number directly, so this ordering no longer affects anything —
# real training/generation reads `elements` from the config or from
# `model.training_elements` persisted on the checkpoint.
ALL_KNOWN_ELEMENTS = ["Ti", "Ge", "O"]

# Batching / threading thresholds used by analysis and neighbor-list
# code paths.
MAX_W = max(1, min(os.cpu_count() or 1, 32))
_LT = 2000          # "large-N" sub-sampling threshold
_CT = 1000          # cell-list vs. brute-force crossover
_CL = 500           # cluster threshold
_BB = 512 << 20     # 512 MiB byte budget


def _atom_color(symbol):
    """Return the jmol_colors RGB tuple for `symbol`, or pink for unknown
    elements.  Used for the legend in `rnd_atoms` so the legend and the
    ASE-drawn atom colors stay in lockstep for any element ASE knows.
    """
    z = ASE_Z.get(symbol)
    if z is None or z < 0 or z >= len(_ASE_JMOL):
        return "#FF69B4"   # only reached for elements ASE doesn't know
    return tuple(float(c) for c in _ASE_JMOL[z])


# Kept for backward compatibility with any external code that may
# still import CPK; new code should call _atom_color() instead.
CPK = {"O": _atom_color("O"), "Ge": _atom_color("Ge"),
       "Ti": _atom_color("Ti"), "default": "#FF69B4"}
