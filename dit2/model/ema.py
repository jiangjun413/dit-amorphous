import copy
import logging
import torch
import torch.nn as nn

# ============================================================
class ModelEMA:
    """
    Exponential moving average of model weights.

    Memory design
    ─────────────
    • shadow: lives on the SAME device as the model (GPU typically) in the
      same dtype as the parameters.  For a ~500K-param fp32 model that is
      ~2 MB VRAM — negligible and eliminates the per-step GPU→CPU transfer
      that used to dominate training throughput.
    • _backup: when apply_to() is called for EMA evaluation, the current
      GPU weights are snapped to CPU before EMA weights replace them on
      the GPU. This keeps peak VRAM at 1× model size during eval.
      restore() copies CPU backup back to GPU and clears the backup dict.
    • update(): iterates a pre-built index of named_parameters +
      named_buffers once, doing an in-place lerp on the GPU tensors.

    DDP note: all ranks maintain identical EMA shadows because DDP
    guarantees identical model weights after each synchronized step.
    Always pass the *unwrapped* model (raw_model) so that state_dict
    keys are portable (no "module." prefix).
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay   = decay
        # Snapshot initial weights on the same device as model.
        # Keep shadow in FP32 for numerical stability — important under AMP
        # where params may be bf16/fp16.  Cost: 2× parameter bytes vs. fp16
        # model, but still trivial (~2-10 MB for typical amorphous-oxide GNN).
        self.shadow  = {}
        for k, v in model.state_dict().items():
            v = v.detach()
            if v.is_floating_point():
                self.shadow[k] = v.float().clone()   # on model device, fp32
            else:
                # Integer buffers (e.g. long-term counters) — keep dtype
                self.shadow[k] = v.clone()
        # Map state_dict key -> padding_idx for every `nn.Embedding` with
        # `padding_idx` set.  Used by update()/apply_to() to keep the
        # padding row pinned at zero in the EMA shadow, matching the live
        # model's behaviour.  Without this, numerical noise (or any future
        # code that writes to the padding row of the live param) could
        # let the shadow drift away from zero.
        self._embed_pad_idx = {}
        for mod_name, mod in model.named_modules():
            if isinstance(mod, nn.Embedding) and mod.padding_idx is not None:
                key = f"{mod_name}.weight" if mod_name else "weight"
                self._embed_pad_idx[key] = int(mod.padding_idx)
        # Pin padding rows immediately so the initial shadow matches.
        for k, pad in self._embed_pad_idx.items():
            if k in self.shadow and self.shadow[k].dim() >= 1:
                self.shadow[k][pad].zero_()
        self._backup = {}
        # `_applied` tracks whether the live model currently holds EMA
        # weights (True) or training weights (False).  Set True only after
        # `apply_to` finishes the swap; cleared by `restore`.  Lets a
        # crash/interrupt path detect "we're mid-swap" and put the model
        # back into a safe state before saving.  Without this flag, an
        # interrupt during the eval-time EMA swap would leave the live
        # model holding EMA weights and `_backup` holding the actual
        # training weights — which `finalize()` would then save as the
        # "main" model, losing the training weights.
        self._applied = False
        # Build (name → param_or_buffer) index for fast update()
        self._named_state = {}
        for name, p in model.named_parameters():
            self._named_state[name] = p
        for name, b in model.named_buffers():
            self._named_state[name] = b

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update shadow weights.  Called once per training step.

        In-place lerp in fp32 on the SAME device as the model — eliminates
        the per-step GPU→CPU transfer that dominated training throughput
        in earlier revisions.  Cast of param→fp32 happens as a fused op
        inside lerp_ (no extra allocation when shapes match).
        """
        one_minus = 1.0 - self.decay
        for name, param in self._named_state.items():
            s = self.shadow.get(name)
            if s is None:
                continue
            if not s.is_floating_point():
                # Integer buffer: follow param directly
                s.copy_(param.data.detach())
                continue
            # lerp in fp32: s = s + (param - s) * (1 - decay)
            # `param.data.detach().float()` is a view/cast; lerp_ fuses it.
            s.lerp_(param.data.detach().float(), one_minus)
        # Force padding rows of EMA-tracked embeddings back to zero AFTER
        # the lerp, matching the live model's `padding_idx=0` semantics.
        # The live model zeros row 0 only at construction; without this
        # pin, a non-zero live row 0 (from warm-start, fp16 underflow,
        # etc.) would slowly leak into the shadow.
        for k, pad in self._embed_pad_idx.items():
            s = self.shadow.get(k)
            if s is not None and s.dim() >= 1 and pad < s.shape[0]:
                s[pad].zero_()

    def apply_to(self, model: nn.Module):
        """
        Copy EMA weights into model for evaluation.

        Backup stays on CPU so peak GPU VRAM stays at 1× model size.
        Shadow is stored in fp32 on the model's device; cast back to
        each target param/buffer's dtype when loading so load_state_dict
        is strict-compatible.

        Sets ``self._applied = True`` only AFTER the swap completes, so
        an interrupt mid-`apply_to` (e.g. partway through the CPU
        backup) leaves the flag at False and the live model still
        holding training weights — the safe state.
        """
        if self._applied:
            # Already swapped — nested apply_to would clobber the
            # training-weights backup with EMA weights, irrecoverably
            # losing them.  Refuse instead.
            raise RuntimeError(
                "ModelEMA.apply_to() called while already applied. "
                "Call restore() first, or check ema.is_applied().")
        self._backup = {}
        state = model.state_dict()
        for k, v in state.items():
            self._backup[k] = v.detach().cpu().clone()
        device = next(model.parameters()).device
        new_state = {}
        for k, v in self.shadow.items():
            target = state.get(k)
            if target is None:
                continue
            new_state[k] = v.to(device=device, dtype=target.dtype)
        model.load_state_dict(new_state, strict=True)
        # Mark applied only after the load_state_dict call returns —
        # this is the point at which the live model genuinely holds
        # EMA weights.  See class-level note on interrupt safety.
        self._applied = True

    def restore(self, model: nn.Module):
        """Restore training weights and immediately free backup.

        Idempotent: calling `restore()` on an EMA that isn't currently
        applied is a no-op.  Lets ``finalize()`` defensively call this
        without first checking state.
        """
        if not self._applied:
            return
        device = next(model.parameters()).device
        model.load_state_dict(
            {k: v.to(device) for k, v in self._backup.items()}, strict=True
        )
        self._backup.clear()   # free CPU backup immediately
        self._applied = False

    def is_applied(self) -> bool:
        """True iff the live model currently holds EMA (not training) weights."""
        return self._applied

    # ------------------------------------------------------------------
    # C3+C4: serialization surface for cross-resume restoration.
    #
    # The training loop saves the bare model as a whole pickle at
    # ``cfg['path_save_model']`` (kept for inference back-compat); the
    # *training state* — optimizer moments, EMA shadow, and the step
    # counter — lives in a sibling bundle saved through `utils/io.py`.
    # See `_save_state_bundle` / `_try_load_state_bundle`.
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        """Return a portable snapshot of the EMA shadow.

        Tensors are detached and moved to CPU so the resulting dict can
        be ``torch.save``d without dragging the model's CUDA context
        along, and so a checkpoint produced on N GPUs is readable on
        any device.  ``decay`` is included for diagnostic purposes —
        the caller's config wins on restore.
        """
        return {
            'shadow': {k: v.detach().cpu().clone()
                       for k, v in self.shadow.items()},
            'decay':  self.decay,
            'embed_pad_idx': dict(self._embed_pad_idx),
        }

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict, strict: bool = False):
        """Restore EMA shadow from a previously-saved ``state_dict()``.

        Keys missing on either side are skipped (lets warm-starts from
        a smaller / differently-structured model proceed gracefully).
        Tensor dtypes and devices are matched to the existing shadow
        slots, not to the saved tensors, so cross-precision and
        cross-device restores work.

        Padding-row pinning is re-applied after the copy in case the
        saved tensors drifted from zero (rare, but possible if the
        bundle came from a code revision that predates the pin).

        Returns ``(n_restored, missing_keys)`` (audit 2026-08-06) so the
        caller can tell a real restore from a silent no-op: with
        ``strict=False`` a key-set change — e.g. resuming a bare graphite
        pickle that then gets wrapped in ``GraphiteModelAdapter``, which
        prefixes every key with ``model.`` — restores ZERO tensors and
        leaves the shadow at its fresh initialisation, while the caller
        happily logs "EMA shadow restored".  Returning a value is purely
        additive; no existing caller inspects it.
        """
        if not isinstance(state_dict, dict):
            return 0, []
        loaded_shadow = state_dict.get('shadow', state_dict)
        if not isinstance(loaded_shadow, dict):
            return 0, []
        missing = []
        for k, dst in self.shadow.items():
            src = loaded_shadow.get(k)
            if src is None:
                missing.append(k); continue
            try:
                dst.copy_(src.to(device=dst.device, dtype=dst.dtype))
            except RuntimeError:
                missing.append(k)
        if strict and missing:
            raise RuntimeError(
                f"ModelEMA.load_state_dict: missing keys {missing!r}")
        # Re-pin padding rows.
        for k, pad in self._embed_pad_idx.items():
            s = self.shadow.get(k)
            if s is not None and s.dim() >= 1 and pad < s.shape[0]:
                s[pad].zero_()
        # The live model still holds training weights — clear any
        # stale `_applied` flag from before the load.
        self._applied = False
        self._backup.clear()
        return len(self.shadow) - len(missing), missing

