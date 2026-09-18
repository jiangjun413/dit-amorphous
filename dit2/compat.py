import datetime
import logging
import os
import sys
import types
import warnings
from pathlib import Path

import torch
import torch.nn as nn

from dit2.constants import ALL_KNOWN_ELEMENTS
from dit2.model import convolutions as _dit_convolutions
from dit2.model import embeddings as _dit_embeddings
from dit2.model import graphite as _dit_graphite
from dit2.model import ema as _dit_ema
from dit2.model.embeddings import BesselBasisRB
from dit2.model.graphite import GraphiteModelAdapter, NequIP_EnergyEmbed

# ── Parent-package checkpoint compat ─────────────────────────────────
# Checkpoints saved by the parent `dit` package are whole-module pickles
# whose classes carry __module__ = 'dit.model.…'.  Alias that hierarchy
# onto dit2's modules so torch.load of parent checkpoints (warm starts,
# calibration baselines) works without the parent package on sys.path.
# A REAL importable `dit` package always wins — we never clobber it.
import sys as _sys
import types as _types
if 'dit' not in _sys.modules:
    try:
        import dit as _real_dit  # noqa: F401 — parent package present
    except ImportError:
        _d = _types.ModuleType('dit')
        _dm = _types.ModuleType('dit.model')
        _d.model = _dm
        for _name, _mod in (('convolutions', _dit_convolutions),
                            ('embeddings', _dit_embeddings),
                            ('graphite', _dit_graphite),
                            ('ema', _dit_ema)):
            setattr(_dm, _name, _mod)
            _sys.modules[f'dit.model.{_name}'] = _mod
        _sys.modules['dit'] = _d
        _sys.modules['dit.model'] = _dm


def _is_adapter(m):
    """GraphiteModelAdapter check tolerant of the PARENT package's class.

    With the parent `dit` installed (pip -e), parent checkpoints unpickle
    as dit.model.graphite.GraphiteModelAdapter — structurally identical but
    a different class object, so dit2's plain isinstance rejects it and
    metadata lookups (training_elements, e_embed patches) silently miss.
    """
    return isinstance(m, GraphiteModelAdapter) or \
        type(m).__name__ == 'GraphiteModelAdapter'


# ---------------------------------------------------------------------
# Legacy-checkpoint pickle compatibility
# ---------------------------------------------------------------------
# The original monolithic ``dit10.py`` saved models with
# ``torch.save(model, …)`` — a *whole-module* pickle that records every
# class's ``__module__`` string at save time.  Checkpoints produced by
# that script therefore reference classes under ``__main__`` (when
# invoked as ``python dit10.py``) or under ``dit10`` (when imported as
# a module).  After this refactor those classes live in
# ``dit.model.{graphite,convolutions,embeddings}``.
#
# Without a compatibility shim, ``torch.load`` on a pre-refactor file
# fails with ``AttributeError: Can't get attribute 'NequIP_EnergyEmbed'
# on <module '__main__'>`` (or ``on <module 'dit10'>``).
#
# We patch the unpickle path by installing two fake module objects in
# ``sys.modules`` — ``__main__`` and ``dit10`` — that expose every
# class name pickled by the original code.  The fake modules are *views*
# over the real new modules: attribute lookup falls through to the
# canonical class in ``dit.model.*``.  This is done unconditionally at
# import time of ``dit.compat`` because checkpoint loading is the only
# thing ``compat`` exists for, and the user-facing entry point always
# imports ``dit.compat`` before ``torch.load`` runs.
#
# The names below are every class pickled into a checkpoint by the
# original script.  Helper functions (``_make_bn`` etc.) are NOT
# included — they're never instantiated, only invoked, so pickle never
# stores a reference to them.

_LEGACY_CLASSES = {
    # main model + adapter
    'NequIP_EnergyEmbed':       _dit_graphite.NequIP_EnergyEmbed,
    'GraphiteModelAdapter':     _dit_graphite.GraphiteModelAdapter,
    '_GraphiteInteraction':     _dit_graphite._GraphiteInteraction,
    '_GraphiteCompose':         _dit_graphite._GraphiteCompose,
    '_GraphiteGaussianBasisEmbedding': _dit_graphite._GraphiteGaussianBasisEmbedding,
    '_GraphiteInitialEmbedding':_dit_graphite._GraphiteInitialEmbedding,
    # convolutions
    'NequIP_MultiConv':         _dit_convolutions.NequIP_MultiConv,
    'StandardConvE3':           _dit_convolutions.StandardConvE3,
    'UVUConv':                  _dit_convolutions.UVUConv,
    'AttnConv':                 _dit_convolutions.AttnConv,
    'UVUAttnConv':              _dit_convolutions.UVUAttnConv,
    'DualConv':                 _dit_convolutions.DualConv,
    'ScalarAttn':               _dit_convolutions.ScalarAttn,
    'LRAttn':                   _dit_convolutions.LRAttn,
    # embeddings
    'InitialEmbedding':         _dit_embeddings.InitialEmbedding,
    'GaussianBasis':            _dit_embeddings.GaussianBasis,
    'ScalarCondEmbed':          _dit_embeddings.ScalarCondEmbed,
    'BesselBasisRB':            _dit_embeddings.BesselBasisRB,
    'EqGate':                   _dit_embeddings.EqGate,
    'RegionEmbed':              _dit_embeddings.RegionEmbed,
    'MethodEmbed':              _dit_embeddings.MethodEmbed,
    'RDFEmbed':                 _dit_embeddings.RDFEmbed,
}


def _install_legacy_module_aliases() -> None:
    """Expose ``_LEGACY_CLASSES`` under ``__main__`` and ``dit10`` so
    ``torch.load`` succeeds on pre-refactor checkpoints.

    Re-entrant and idempotent: re-running is safe and cheap.  We never
    *replace* a real ``__main__`` module — we attach our compat
    attributes onto whatever ``sys.modules['__main__']`` already is
    (typically the user's entry script).  For ``dit10`` we install a
    fresh ``types.ModuleType`` if it isn't already present.
    """
    main_mod = sys.modules.get('__main__')
    if main_mod is None:
        main_mod = types.ModuleType('__main__')
        sys.modules['__main__'] = main_mod
    for name, cls in _LEGACY_CLASSES.items():
        if not hasattr(main_mod, name):
            setattr(main_mod, name, cls)

    dit10_mod = sys.modules.get('dit10')
    if dit10_mod is None:
        dit10_mod = types.ModuleType('dit10')
        sys.modules['dit10'] = dit10_mod
    for name, cls in _LEGACY_CLASSES.items():
        if not hasattr(dit10_mod, name):
            setattr(dit10_mod, name, cls)


_install_legacy_module_aliases()


def _compat_patch_energy_embed(model):
    """Fix old NequIP_EnergyEmbed checkpoints that predate Fix 1.

    Before Fix 1, e_projection mapped to irreps.dim (all channels).
    After Fix 1, it maps to scalar_dim only (l=0 channels) with zero-padding.

    torch.load restores __dict__ directly, so old checkpoints lack
    _scalar_dim and _irreps_full_dim.  If those attrs are absent we infer
    them from the saved e_projection weight shape:
      - If out_features == irreps.dim  → old checkpoint, pad needed
      - If out_features == scalar_dim  → already patched, nothing to do
    We set the attributes so forward() works correctly either way.
    """
    inner = model.model if _is_adapter(model) else model
    if not (isinstance(inner, NequIP_EnergyEmbed)
            or type(inner).__name__ == 'NequIP_EnergyEmbed'):
        return model
    if hasattr(inner, '_scalar_dim') and hasattr(inner, '_irreps_full_dim'):
        return model  # already correct (new checkpoint or previously patched)
    # Infer dims from e_projection's final Linear layer.
    # Defensive: older / future variants might wrap e_projection in a
    # non-Sequential (single Linear) — accept both shapes rather than
    # raising on `[-1]` indexing.
    e_proj = inner.e_projection
    last_lin = e_proj[-1] if hasattr(e_proj, '__getitem__') else e_proj
    if not hasattr(last_lin, 'out_features'):
        # Cannot infer — leave the model untouched and hope the forward
        # path's own guards handle it.  Better than crashing on load.
        return model
    proj_out = last_lin.out_features  # weight shape (out, in)
    # Compute scalar_dim and full_dim from the last interaction's irreps_out
    # (gate.irreps_out after the last conv layer)
    last_gate = inner.interactions[-1].second  # _GraphiteCompose.second = Gate
    irreps_out = last_gate.irreps_out
    scalar_dim  = sum(mul * (2 * ir.l + 1) for mul, ir in irreps_out if ir.l == 0)
    full_dim    = irreps_out.dim
    inner._irreps_full_dim = full_dim
    if proj_out == full_dim:
        # Old checkpoint: e_projection outputs full irreps.dim.
        # We cannot magically shrink the weight, but we CAN make forward()
        # aware that no zero-padding is needed (old behaviour preserved).
        # Set _scalar_dim = full_dim so the pad_dim branch is skipped.
        inner._scalar_dim = full_dim
    else:
        # New checkpoint with scalar_dim output — just set the attribute.
        inner._scalar_dim = scalar_dim
    return model


def _compat_patch_dual_cutoff(model):
    """Backfill missing attrs on old NequIP_MultiConv dual_cutoff checkpoints.

    Older dit3 revisions added `self.long_range_cutoff` and `self.sr_cutoff`
    on the model, and `sr_cut` on LRAttn for the smooth envelope.  Models
    saved BEFORE those revisions won't have these attrs, so:
      - _detect_dual_cutoff() returns None → VL won't use lr_cutoff
      - LRAttn.forward() won't apply the smooth envelope

    We detect dual_cutoff by presence of a `lr_rbf` BesselBasisRB module
    and recover both cutoffs from its `.cutoff` and from the inner conv
    `rbf.cutoff`.
    """
    inner = model.model if _is_adapter(model) else model
    ct = getattr(inner, 'conv_type', None)

    # ── Dual-cutoff-specific attrs (sr_cutoff, long_range_cutoff, LRAttn.sr_cut) ──
    if ct == 'dual_cutoff':
        if not hasattr(inner, 'long_range_cutoff'):
            lr_rbf = getattr(inner, 'lr_rbf', None)
            if lr_rbf is not None and hasattr(lr_rbf, 'cutoff'):
                inner.long_range_cutoff = float(lr_rbf.cutoff)
        if not hasattr(inner, 'sr_cutoff'):
            sr_rbf = getattr(inner, 'rbf', None)
            if sr_rbf is not None and hasattr(sr_rbf, 'cutoff'):
                inner.sr_cutoff = float(sr_rbf.cutoff)
            else:
                inner.sr_cutoff = getattr(inner, 'cutoff', None)
        sr = getattr(inner, 'sr_cutoff', None)
        if sr is not None:
            for conv in getattr(inner, 'convs', []):
                lr_attn = getattr(conv, 'lr_attn', None)
                if lr_attn is not None and getattr(lr_attn, 'sr_cut', None) is None:
                    lr_attn.sr_cut = float(sr)
                if lr_attn is not None and not hasattr(lr_attn, 'lr_cut'):
                    lr_attn.lr_cut = float(inner.long_range_cutoff)

    # ── Stability-fix backfill for ALL conv types ──
    # standard_e3 / uvu / attention / uvu_attention / dual_cutoff all grew
    # a new `alpha_sr` scalar parameter and (for dual) `alpha_lr` + `nn_`
    # in this revision.  Older checkpoints lack these; we attach them with
    # values that RECOVER THE PRE-FIX BEHAVIOUR so existing weights still
    # give the same forward output:
    #     alpha_sr = 1.0  → conv contribution flows through unchanged
    #     alpha_lr = 1.0  → LR contribution flows through unchanged
    #     nn_      = float(num_neighbors)
    # If the user wants the new stabilised dynamics (with density scaling
    # and SkipInit ramp-on), they should retrain from scratch — loading
    # old weights with alpha=0 would effectively mute the conv and break
    # inference.
    default_nn = float(getattr(inner, 'num_neighbors', 12))
    for conv in getattr(inner, 'convs', []):
        # Any conv with a `tp` FullyConnectedTensorProduct is a candidate
        if not hasattr(conv, 'tp'):
            continue
        if not hasattr(conv, 'alpha_sr'):
            conv.alpha_sr = nn.Parameter(torch.ones(1))
        # alpha_lr only exists on DualConv
        is_dual_conv = hasattr(conv, 'lr_attn')
        if is_dual_conv and not hasattr(conv, 'alpha_lr'):
            conv.alpha_lr = nn.Parameter(torch.ones(1))
        # nn_ is needed on DualConv and on the attention variants for
        # √-degree scaling.  Pre-fix StandardConvE3/UVUConv also had nn_
        # (via self.nn_ = nn_), so this only affects Attn/UVUAttn/Dual.
        if not hasattr(conv, 'nn_'):
            conv.nn_ = default_nn
    return model


def load_model_raw(path):
    m = torch.load(path, map_location="cpu", weights_only=False).cpu()
    for p in m.parameters(): p.data = p.data.cpu()
    for b in m.buffers(): b.data = b.data.cpu()
    # Auto-wrap graphite-style NequIP_EnergyEmbed in adapter so that
    # forward(data) works (NequIP_EnergyEmbed.forward expects two args:
    # data + energy, but generate() and loss_fn() call model(data)).
    if hasattr(m, 'init_embed') and not _is_adapter(m):
        m = GraphiteModelAdapter(m)
    # Patch missing Fix-1 attributes on old checkpoints
    m = _compat_patch_energy_embed(m)
    # Patch missing dual_cutoff attributes on older NequIP_MultiConv checkpoints
    m = _compat_patch_dual_cutoff(m)
    m.eval(); return m

def get_mi(model):
    info = {}
    for mod in model.modules():
        # Try legacy graphite-style name, then modern NequIP_MultiConv name.
        emb = getattr(mod, "embed_node_x", None) or getattr(mod, "embed_x", None)
        if isinstance(emb, nn.Embedding):
            info["Element types (n_species)"] = str(emb.num_embeddings)
            info["Node embedding dim"] = str(emb.embedding_dim); break
    # Fallback: first nn.Embedding (last-resort heuristic)
    if "Element types (n_species)" not in info:
        for mod in model.modules():
            if isinstance(mod, nn.Embedding):
                info["Element types (n_species)"] = str(mod.num_embeddings)
                info["Node embedding dim"] = str(mod.embedding_dim); break
    # Detect GNN cutoff from various model types
    cutoff_found = False
    for mod in model.modules():
        # NequIP_MultiConv stores cutoff directly
        if hasattr(mod, 'cutoff') and isinstance(getattr(mod, 'cutoff', None), (int, float)):
            info["GNN cutoff (A)"] = str(mod.cutoff)
            cutoff_found = True; break
    if not cutoff_found:
        for mod in model.modules():
            # BesselBasisRB (used in NequIP_MultiConv)
            if isinstance(mod, BesselBasisRB):
                info["GNN cutoff (A)"] = str(mod.cutoff)
                cutoff_found = True; break
    if not cutoff_found:
        for mod in model.modules():
            # _GraphiteInitialEmbedding / InitialEmbedding
            c = getattr(mod, '_cutoff', None)
            if c is not None:
                info["GNN cutoff (A)"] = str(c)
                cutoff_found = True; break
    if not cutoff_found:
        # Legacy: embed_edge with keywords
        for mod in model.modules():
            ee = getattr(mod, "embed_edge", None)
            if ee:
                kw = getattr(ee, "keywords", {})
                if "end" in kw: info["GNN cutoff (A)"] = str(kw["end"])
                if "num_basis" in kw: info["Bessel basis"] = str(kw["num_basis"])
                break
    info["Total parameters"] = f"{sum(p.numel() for p in model.parameters()):,}"
    info["Model class"] = type(model).__name__
    return info

def get_model_elements(model, cfg_elements=None):
    """Determine the model's element list.

    Priority:
      1. Persisted training_elements on the model (always correct, reflects
         what the encoder actually saw during training)
      2. cfg_elements (from config model.elements)
      3. Detect n_species from embedding, then validate against cfg_elements

    Returns (element_list, source_str) so callers can log the source.
    """
    # Detect n_species from model
    n_sp = None
    for mod in model.modules():
        # Legacy graphite-style or modern NequIP_MultiConv.
        emb = getattr(mod, "embed_node_x", None) or getattr(mod, "embed_x", None)
        if isinstance(emb, nn.Embedding):
            n_sp = emb.num_embeddings; break
    if n_sp is None:
        # Fallback: first nn.Embedding
        for mod in model.modules():
            if isinstance(mod, nn.Embedding):
                n_sp = mod.num_embeddings; break

    # ── Priority 1: persisted training_elements (authoritative) ──
    # When the model has a persisted training_elements list, that list is
    # authoritative — it reflects the exact index ordering the encoder
    # learned. Any config-supplied list (cfg_elements) is silently
    # overridden; callers (e.g. the Streamlit sidebar) sync their input
    # widget to the returned list so the UI shows the values actually used.
    inner = model.model if _is_adapter(model) else model
    saved = getattr(inner, 'training_elements', None)
    if saved is not None and len(saved) > 0:
        return list(saved), "model.training_elements"

    # If config specifies elements, use them (validated against model)
    if cfg_elements and len(cfg_elements) > 0:
        if n_sp is not None and len(cfg_elements) != n_sp:
            raise ValueError(
                f"config model.elements has {len(cfg_elements)} elements {cfg_elements} "
                f"but model embedding has n_species={n_sp}. They must match.")
        return list(cfg_elements), "config"

    # No config — guess from n_species and ALL_KNOWN_ELEMENTS
    if n_sp is None:
        return list(ALL_KNOWN_ELEMENTS), "fallback(no embedding found)"

    if n_sp == len(ALL_KNOWN_ELEMENTS):
        return list(ALL_KNOWN_ELEMENTS), "auto(all)"

    # n_sp < len(ALL_KNOWN_ELEMENTS): ambiguous — we don't know WHICH elements
    # Common cases: 2-element models are typically binary oxides (XO₂)
    # Raise error asking user to specify
    raise ValueError(
        f"Model has n_species={n_sp} but {len(ALL_KNOWN_ELEMENTS)} known elements "
        f"{ALL_KNOWN_ELEMENTS}. Cannot auto-detect which {n_sp} elements the model uses.\n"
        f"Please set model.elements in config.yaml, e.g.:\n"
        f"  model:\n"
        f"    elements: [Ge, O]    # for a GeO₂ model\n"
        f"    elements: [Ti, O]    # for a TiO₂ model")

def mi_text(info, path):
    return "\n".join(["="*60, f"MODEL: {path}", f"Time: {datetime.datetime.now()}", "="*60] +
                     [f"  {k:<30s}: {v}" for k, v in info.items()])
