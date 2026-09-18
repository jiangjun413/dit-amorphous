import configparser
import datetime
from pathlib import Path

def _get(cp, section, key, fallback=None):
    return cp.get(section, key, fallback=fallback)

def _getint(cp, section, key, fallback=None):
    raw = _get(cp, section, key)
    return int(raw) if raw is not None else fallback

def _getfloat(cp, section, key, fallback=None):
    raw = _get(cp, section, key)
    return float(raw) if raw is not None else fallback

def _getbool(cp, section, key, fallback=None):
    raw = _get(cp, section, key)
    if raw is None: return fallback
    return raw.strip().lower() in ('true', '1', 'yes', 'on')

def _getfloat_or_none(cp, section, key):
    raw = _get(cp, section, key, fallback='').strip()
    return float(raw) if raw else None

def _getintlist(cp, section, key, fallback=None):
    raw = _get(cp, section, key, fallback='').strip()
    return [int(x) for x in raw.split()] if raw else fallback

def _getstr_or_none(cp, section, key):
    raw = _get(cp, section, key, fallback='').strip()
    return raw if raw else None

def load_config(path: str) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Config file not found: '{path}'\n"
            "Create one from the template or pass --config <path>."
        )
    cp = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    cp.read(path)

    # ── [run] ──────────────────────────────────────────────
    gpu             = _getint        (cp, 'run', 'gpu',             fallback=0)
    resume          = _getbool       (cp, 'run', 'resume',          fallback=True)
    path_load_model = _getstr_or_none(cp, 'run', 'path_load_model')
    path_save_model = _get           (cp, 'run', 'path_save_model', fallback='./model.pt')
    path_save_loss  = _get           (cp, 'run', 'path_save_loss',  fallback='./loss.png')
    save_every      = _getint        (cp, 'run', 'save_every',      fallback=5000)
    log_path        = _getstr_or_none(cp, 'run', 'log_path')

    # ── [data] ─────────────────────────────────────────────
    num_sources  = _getint(cp, 'data', 'num_sources', fallback=1)
    # ΔE ceiling screen (ported from dit6 2026-08-23).  None/absent = OFF, so
    # every pre-existing dit2 config is unaffected.  1.0 matches the
    # delta_e_min conditioning ceiling in SCALAR_COND_DEFAULTS.
    _mde = _get(cp, 'data', 'max_delta_e', fallback=None)
    max_delta_e = (float(_mde)
                   if _mde not in (None, '', 'none', 'None') else None)
    # M8: run-level energy_key default; per-source override via
    # `source_<n>_energy_key`.  Choices: auto | E0 | free | total.
    # See VASPDataExtractor for what each means.  Default 'auto'
    # preserves legacy behaviour (silent E0/free mix across file types).
    # ── OPT-IN pair-aware physicality screen (config: [data] physicality_gate)
    # Path to a pair_gate_table.json (docs/pair_gate.py).  Absent = OFF, so every
    # pre-existing dit2 run stays BIT-IDENTICAL.  See dit2/data/physicality.py for
    # why this replaces the global `dmin >= 1.45 A` rule that every campaign used.
    _pg = _get(cp, 'data', 'physicality_gate', fallback=None)
    physicality_gate = (None if _pg in (None, '', 'none', 'None')
                        else str(_pg).strip())
    physicality_reject = _get(cp, 'data', 'physicality_reject',
                              fallback='overlap').strip().lower()
    if physicality_reject not in ('overlap', 'overlap+contact'):
        raise ValueError(
            f"[data] physicality_reject must be 'overlap' or 'overlap+contact', "
            f"got {physicality_reject!r}.")
    energy_key = _get(cp, 'data', 'energy_key', fallback='auto').strip()
    if energy_key not in ('auto', 'E0', 'free', 'total'):
        raise ValueError(
            f"[data] energy_key must be one of "
            f"('auto', 'E0', 'free', 'total'), got {energy_key!r}.")
    data_sources = []
    for n in range(num_sources):
        data_dir      = _get   (cp, 'data', f'source_{n}_data_dir')
        config_select = _getint(cp, 'data', f'source_{n}_config_select', fallback=-1)
        pickle_path   = _get   (cp, 'data', f'source_{n}_pickle_path')
        src_energy    = _get   (cp, 'data', f'source_{n}_energy_key', fallback=None)
        if not data_dir or not pickle_path:
            raise ValueError(
                f"[data] source_{n}_data_dir and source_{n}_pickle_path "
                f"are both required when num_sources={num_sources}."
            )
        src = dict(data_dir=data_dir, config_select=config_select,
                   pickle_path=pickle_path)
        if src_energy is not None:
            src_energy = src_energy.strip()
            if src_energy not in ('auto', 'E0', 'free', 'total'):
                raise ValueError(
                    f"[data] source_{n}_energy_key must be one of "
                    f"('auto', 'E0', 'free', 'total'), got "
                    f"{src_energy!r}.")
            src['energy_key'] = src_energy
        # Conditioning v2 per-source overrides (optional): a simulation-method
        # label and a whole-directory temperature when the path lacks a K-token.
        src_method = _get(cp, 'data', f'source_{n}_method', fallback=None)
        if src_method:
            src['method'] = src_method.strip()
        src_temp = _get(cp, 'data', f'source_{n}_temperature', fallback=None)
        if src_temp is not None and src_temp.strip():
            src['temperature'] = float(src_temp)
        data_sources.append(src)

    # ── [training] ─────────────────────────────────────────
    # `num_species` and `elements` used to live here.  Both are gone:
    # the model embeds every Z = 0..118 in a fixed periodic-table-wide
    # table (see `NequIP_MultiConv.PERIODIC_TABLE_SIZE = 119`), so neither
    # the vocab size nor the symbol list affects the model definition.
    # The element set the model was actually trained on is auto-detected
    # from the data and stamped on the checkpoint as `model.training_z`
    # / `model.training_elements`.
    batch_size       = _getint        (cp, 'training', 'batch_size',       fallback=64)
    num_updates      = _getint        (cp, 'training', 'num_updates',      fallback=100_000)
    time_limit_hours = _getfloat_or_none(cp, 'training', 'time_limit_hours')
    learn_rate       = _getfloat      (cp, 'training', 'learn_rate',       fallback=2e-4)
    weight_decay     = _getfloat      (cp, 'training', 'weight_decay',     fallback=1e-5)
    grad_clip        = _getfloat      (cp, 'training', 'grad_clip',        fallback=1.0)
    duplicate        = _getint        (cp, 'training', 'duplicate',        fallback=128)
    val_duplicate    = _getint        (cp, 'training', 'val_duplicate',    fallback=4)
    train_ratio      = _getfloat      (cp, 'training', 'train_ratio',      fallback=0.9)
    data_split_seed  = _getint        (cp, 'training', 'data_split_seed',  fallback=42)
    sigma_max        = _getfloat      (cp, 'training', 'sigma_max',        fallback=0.75)
    eta_window       = _getint        (cp, 'training', 'eta_window',       fallback=200)
    use_amp          = _getbool       (cp, 'training', 'use_amp',          fallback=True)
    ema_decay        = _getfloat      (cp, 'training', 'ema_decay',        fallback=0.999)
    # How often (in steps) to all-reduce the early-stop flag across DDP ranks.
    # 1 = check every step (legacy behaviour, expensive: NCCL collective per
    # iteration).  Higher values amortise the collective cost; trade-off is
    # that a local SIGTERM may take up to N-1 steps to be globally observed.
    # Single-GPU runs ignore this setting (no collective).
    stop_check_every = _getint        (cp, 'training', 'stop_check_every', fallback=50)
    # σ-weighting for the displacement-prediction loss.
    #   'none'         — unweighted MSE on dx (legacy; default for back-compat).
    #                    Gradients dominated by large-σ samples since
    #                    var(dx)=σ² ranges over ~6 orders of magnitude.
    #   'inv_sigma_sq' — multiply elementwise squared error by 1/σ² per
    #                    graph, equivalent to ε-prediction MSE.  Brings
    #                    small-σ samples back into the gradient signal.
    loss_sigma_weighting = _get(cp, 'training', 'loss_sigma_weighting',
                                fallback='none').strip().lower()
    if loss_sigma_weighting not in ('none', 'inv_sigma_sq', 'min_snr_gamma'):
        raise ValueError(
            f"[training] loss_sigma_weighting must be 'none', "
            f"'inv_sigma_sq', or 'min_snr_gamma'; "
            f"got {loss_sigma_weighting!r}.")
    # γ for the Min-SNR-γ weight.  Only consulted when
    # loss_sigma_weighting='min_snr_gamma'; read unconditionally so
    # the value appears in the saved config bundle.
    loss_min_snr_gamma = _getfloat(cp, 'training', 'loss_min_snr_gamma',
                                   fallback=5.0)
    if loss_sigma_weighting == 'min_snr_gamma' and loss_min_snr_gamma <= 0:
        raise ValueError(
            f"[training] loss_min_snr_gamma must be > 0; "
            f"got {loss_min_snr_gamma!r}.")
    # Append the σ-binned TRAINING loss to each epoch row (extra columns to
    # the right of ETA; see trainer.SigmaBinStats / SIGMA_BIN_EDGES).  Costs
    # one all-reduce per epoch and two tiny index_add_ kernels per step —
    # default ON.  Set false to restore the pre-2026-08-06 row exactly.
    log_sigma_bins = _getbool(cp, 'training', 'log_sigma_bins', fallback=True)
    # Audit finding F4 (2026-08-06): how often to all-reduce the global
    # "some rank saw a non-finite loss/gradient" flag and skip the update.
    #   1  (default) — every step.  The ONLY setting that protects the
    #                  WEIGHTS: once a NaN gradient is applied the model,
    #                  the EMA and the next checkpoint are all poisoned.
    #   N>1          — throughput escape hatch; only the checkpoint is
    #                  still protected (via the refusal in
    #                  trainer._save_checkpoint).
    #   0 / negative — disabled entirely (not recommended).
    # Cost measured at ~44 us/step (gloo) vs a 33.85 ms live step: ~0.13 %.
    nan_guard_every = _getint(cp, 'training', 'nan_guard_every', fallback=1)

    # ── [model] ────────────────────────────────────────────
    # ── Element embedding ──
    # The whole periodic table (Z = 1..118, with Z=0 reserved as padding_idx)
    # is embedded into one shared latent space of size `element_latent_dim`.
    # This decouples the model from the specific subset of elements in any
    # one dataset — a checkpoint trained on {Ta,Zr,Si,...} can be fine-tuned
    # on a different element set without architectural changes (only the
    # warm-start row remapping in scripts/warm_start.py).
    #
    # `irreps_node_x` and `irreps_node_z` are now DERIVED from
    # `element_latent_dim` and forced to `{element_latent_dim}x0e`.  If old
    # configs still set them, we log a deprecation warning and override.
    element_latent_dim = _getint(cp, 'model', 'element_latent_dim', fallback=8)
    irreps_node_x  = f'{element_latent_dim}x0e'
    irreps_node_z  = f'{element_latent_dim}x0e'
    _legacy_irx = _get(cp, 'model', 'irreps_node_x', fallback=None)
    _legacy_irz = _get(cp, 'model', 'irreps_node_z', fallback=None)
    # Only warn when a legacy value DISAGREES with the derived one —
    # otherwise stale-but-matching configs spam the warning on every run.
    _stale = (_legacy_irx is not None
              and _legacy_irx.strip() != irreps_node_x) or \
             (_legacy_irz is not None
              and _legacy_irz.strip() != irreps_node_z)
    if _stale:
        import warnings
        warnings.warn(
            "[model] irreps_node_x / irreps_node_z are deprecated and "
            f"now derived from element_latent_dim={element_latent_dim} "
            f"(→ '{irreps_node_x}').  Your config sets them to a "
            f"different value (irreps_node_x={_legacy_irx!r}, "
            f"irreps_node_z={_legacy_irz!r}) which will be ignored.",
            DeprecationWarning, stacklevel=2)
    irreps_hidden  = _get       (cp, 'model', 'irreps_hidden',  fallback='64x0e + 32x1e')
    irreps_edge    = _get       (cp, 'model', 'irreps_edge',    fallback='4x0e + 4x1e + 2x2e')
    irreps_out     = _get       (cp, 'model', 'irreps_out',     fallback='1x1e')
    num_convs      = _getint    (cp, 'model', 'num_convs',      fallback=3)
    radial_neurons = _getintlist(cp, 'model', 'radial_neurons', fallback=[16, 64])
    num_neighbors  = _getint    (cp, 'model', 'num_neighbors',  fallback=12)
    # Per-atom region conditioning for interface / heterostructure models
    # (graphite conv_type only).  0 → plain bulk model.  N → the model
    # learns an additive scalar bias per region id 0..N-1 (see RegionEmbed);
    # training data must attach a `region` label array to each graph.
    n_regions      = _getint    (cp, 'model', 'n_regions',     fallback=0)
    # Conditioning v2: categorical simulation-method embedding.  0 → no method
    # tag (legacy).  N → a zero-init additive scalar bias per method id 0..N-1
    # (see MethodEmbed); the data pipeline attaches a per-graph `method` id.
    n_methods      = _getint    (cp, 'model', 'n_methods',     fallback=0)
    # Opt-in partial warm start: build the model fresh from this config, then
    # copy name+shape-matching weights from path_load_model (new heads stay
    # zero-init).  Use when the checkpoint's architecture differs from this
    # config (e.g. adding method/ΔE/RDF conditioning to an energy-only model).
    warm_start_partial = _getbool(cp, 'model', 'warm_start_partial', fallback=False)
    large_cutoff   = _getfloat  (cp, 'model', 'large_cutoff',  fallback=10.0)
    cutoff         = _getfloat  (cp, 'model', 'cutoff',        fallback=5.0)
    # ── Conv architecture ──
    conv_type               = _get    (cp, 'model', 'conv_type',               fallback='uvu')
    # Validate up front: an unrecognised conv_type used to fall through
    # NequIP_MultiConv's dispatch to the StandardConvE3 `else` branch,
    # silently training the wrong architecture (and dropping the entire
    # long-range branch for would-be 'dual_cutoff' runs) while stamping
    # the bogus string on the checkpoint so it still looked right.
    _valid_conv = ('uvu', 'attention', 'uvu_attention', 'dual_cutoff',
                   'standard_e3', 'graphite')
    if conv_type not in _valid_conv:
        raise ValueError(
            f"[model] conv_type must be one of {_valid_conv}; "
            f"got {conv_type!r}.")
    n_radial_basis          = _getint (cp, 'model', 'n_radial_basis',          fallback=16)
    uvu_bottleneck_factor   = _getint (cp, 'model', 'uvu_bottleneck_factor',   fallback=4)
    n_attention_heads       = _getint (cp, 'model', 'n_attention_heads',       fallback=4)
    long_range_cutoff       = _getfloat(cp, 'model', 'long_range_cutoff',     fallback=10.0)
    n_radial_basis_lr       = _getint (cp, 'model', 'n_radial_basis_lr',      fallback=12)

    # ── [conditions] — multi-property conditioning ─────────
    # Space-separated list of property names to condition on.
    # Each must exist in SCALAR_COND_DEFAULTS or have explicit vmin/vmax.
    # dit2: an ABSENT key defaults to the MINIMAL orthogonal recipe —
    # 'delta_e_min' alone.  Absolute energy and T_eff are deterministic
    # functions of (composition, ΔE) and training all three lets the model
    # split the conditioning signal across redundant axes (measured: 3×
    # noisier per-structure energy in generation vs an always-present single
    # axis).  'energy' remains available for legacy-style runs.  An
    # explicitly EMPTY value (``scalar_conditions =``) or the keyword 'none'
    # trains a model with NO scalar conditioning at all.
    _sc_raw = _get(cp, 'conditions', 'scalar_conditions', fallback='delta_e_min').strip()
    scalar_cond_names = [] if _sc_raw.lower() == 'none' else _sc_raw.split()
    # `energy` is always populated.  `temperature` and `delta_e_min` are
    # populated end-to-end by the Conditioning-v2 pipeline (provenance parse +
    # ΔE-basin reduction in dit/data/vasp.py, threaded through the graph
    # builder / dataset).  Any name outside this set would build a Gaussian-
    # basis + MLP branch that never receives a value — silently untrained on
    # single-GPU and a DDP unused-parameter error on multi-GPU — so it is
    # refused up front.
    _supported_cond = {'energy', 'temperature', 'delta_e_min', 'density'}
    _unsupported = [n for n in scalar_cond_names if n not in _supported_cond]
    if _unsupported:
        raise ValueError(
            f"[conditions] scalar_conditions {_unsupported} are not "
            f"supplied by the data pipeline (only {sorted(_supported_cond)} "
            f"are). Conditioning on them would never train. Remove them or "
            f"extend the dataset/graph builder to store them.")

    # ── [conditions] — Conditioning v2 (method tag, ΔE, RDF) ──
    # Method taxonomy: ordered vocabulary mapping method name -> embedding id.
    _tax_raw = _get(cp, 'conditions', 'method_taxonomy',
                    fallback='pbe scan hse mace classical').strip()
    method_taxonomy = _tax_raw.split() if _tax_raw else ['pbe']
    # Which inherent-structure reference the active delta_e_min uses.
    # ── PER-CONDITION BASIS RANGE OVERRIDE (opt-in, 2026-08-25) ────────────
    # `SCALAR_COND_DEFAULTS` is a module-level dict, so widening a range there
    # would change the basis for EVERY from-scratch dit2 build, not just this
    # one.  These keys override it per run instead:
    #
    #     cond_range_density     = 1.3 13.8 48
    #     cond_range_delta_e_min = 0 1.2 40
    #
    # WHY THIS IS NEEDED AT ALL.  The defaults clip real data.  Measured over
    # full + al5 + full3 + heat + solar (14,320 structures): density spans
    # 1.3598-13.6956 g/cc, so the default vmax of 10.0 SATURATES 1.41% of the
    # corpus (202 structures: 153 crystal, 12 relax, 36 heat, 1 solar-Mo).
    # A saturated channel is indistinguishable from any other saturated value,
    # so those structures train against a density label that carries no
    # information.
    #
    # Resumed runs are unaffected either way: GaussianBasis registers `centers`
    # and `width` as BUFFERS, so a checkpoint carries its own trained basis and
    # dit2 resumes by unpickling the module itself.
    cond_ranges = {}
    for _k, _v in cp.items('conditions') if cp.has_section('conditions') else []:
        if not _k.startswith('cond_range_'):
            continue
        _name = _k[len('cond_range_'):]
        _parts = _v.split()
        if len(_parts) != 3:
            raise ValueError(
                f"[conditions] {_k} must be 'vmin vmax n_basis', got {_v!r}")
        _lo, _hi, _nb = float(_parts[0]), float(_parts[1]), int(_parts[2])
        if not (_hi > _lo) or _nb < 2:
            raise ValueError(
                f"[conditions] {_k}: need vmax > vmin and n_basis >= 2, "
                f"got vmin={_lo} vmax={_hi} n_basis={_nb}")
        cond_ranges[_name] = {'vmin': _lo, 'vmax': _hi, 'n_basis': _nb}

    delta_e_ref = _get(cp, 'conditions', 'delta_e_ref', fallback='vasp').strip()
    delta_e_floor = _get(cp, 'conditions', 'delta_e_floor',
                        fallback='inherent').strip().lower()
    if delta_e_floor not in ('inherent', 'crystal'):
        raise ValueError(
            f"[conditions] delta_e_floor must be 'inherent' or 'crystal', "
            f"got {delta_e_floor!r}.")
    if delta_e_ref not in ('vasp', 'mace'):
        raise ValueError(
            f"[conditions] delta_e_ref must be 'vasp' or 'mace', got {delta_e_ref!r}.")
    # Temperature conditioning source.  'eff' (default) = effective
    # temperature derived from ΔE via equipartition (T_eff = ΔE / 1.5 k_B) —
    # defined for EVERY structure.  'path' = the set-point parsed from the
    # directory name (only the md/melt minority carries one; NaN frames fall
    # back to T_eff so the conditioning value is never NaN).
    temperature_mode = _get(cp, 'conditions', 'temperature_mode', fallback='eff').strip()
    if temperature_mode not in ('eff', 'path'):
        raise ValueError(
            f"[conditions] temperature_mode must be 'eff' or 'path', "
            f"got {temperature_mode!r}.")
    # Per-sample probability of turning each conditioning axis OFF during
    # training (classifier-free-guidance style; see CondPresenceDropout).
    # Makes generation with any condition subset — including none —
    # in-distribution.  0 = legacy always-conditioned training.
    cond_dropout = _getfloat(cp, 'conditions', 'cond_dropout', fallback=0.0)
    if not (0.0 <= cond_dropout <= 1.0):
        raise ValueError(f"[conditions] cond_dropout must be in [0,1], "
                         f"got {cond_dropout}.")
    # Structured ("solo") dropout probability over the redundant
    # energy/temperature/delta_e_min trio: with this per-sample probability
    # exactly ONE of those axes (uniform choice) stays on and the other two
    # are forced off, so each axis learns to carry the conditioning signal
    # alone (see CondPresenceDropout.solo_group).  0 = off (legacy).
    cond_solo_prob = _getfloat(cp, 'conditions', 'cond_solo_prob', fallback=0.0)
    if not (0.0 <= cond_solo_prob <= 1.0):
        raise ValueError(f"[conditions] cond_solo_prob must be in [0,1], "
                         f"got {cond_solo_prob}.")
    # Per-condition importance weights (train-time; generation mirrors them
    # via physics.cond_weights).  Space-separated ``name:value`` pairs over
    # {energy, temperature, delta_e_min, method, rdf}; anything omitted
    # defaults to 1.0.  A kept condition's head output is scaled by its
    # weight (the presence mask carries the value), so >1 amplifies an
    # axis's influence and <1 attenuates it.
    _cw_raw = _get(cp, 'conditions', 'cond_weights', fallback='').strip()
    cond_weights = {}
    _valid_cw = ('energy', 'temperature', 'delta_e_min', 'method', 'rdf')
    for tok in (_cw_raw.split() if _cw_raw else []):
        if ':' not in tok:
            raise ValueError(f"[conditions] cond_weights entries must be "
                             f"'name:value', got {tok!r}.")
        k, v = tok.split(':', 1)
        if k not in _valid_cw:
            raise ValueError(f"[conditions] cond_weights name {k!r} invalid; "
                             f"choose from {_valid_cw}.")
        v = float(v)
        if v < 0:
            raise ValueError(f"[conditions] cond_weights[{k}] must be >= 0, "
                             f"got {v}.")
        cond_weights[k] = v
    # RDF conditioning: grid + which weighted-total channels are active.
    rdf_rmax   = _getfloat(cp, 'conditions', 'rdf_rmax',  fallback=8.0)
    rdf_nbins  = _getint  (cp, 'conditions', 'rdf_nbins', fallback=100)
    _rdf_ch_raw = _get(cp, 'conditions', 'rdf_channels', fallback='').strip()
    _valid_rdf_ch = ('number', 'xray', 'electron', 'neutron', 'partial')
    rdf_channels = _rdf_ch_raw.split() if _rdf_ch_raw else []
    _bad_ch = [c for c in rdf_channels if c not in _valid_rdf_ch]
    if _bad_ch:
        raise ValueError(
            f"[conditions] rdf_channels {_bad_ch} invalid; choose from "
            f"{_valid_rdf_ch}.")
    # A v2 provenance read is needed whenever a v2 condition is active.
    provenance = bool(n_methods > 0 or rdf_channels
                      or {'temperature', 'delta_e_min'} & set(scalar_cond_names))

    # ── [performance] ──────────────────────────────────────
    num_workers      = _getint (cp, 'performance', 'num_workers',      fallback=2)
    num_proc_workers = _getint (cp, 'performance', 'num_proc_workers', fallback=8)
    pin_memory       = _getbool(cp, 'performance', 'pin_memory',       fallback=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not log_path:
        log_path = f'./train_{timestamp}.log'

    return dict(
        # run
        gpu=gpu, resume=resume,
        path_load_model=path_load_model, path_save_model=path_save_model,
        path_save_loss=path_save_loss,
        save_every=save_every, log_path=log_path, timestamp=timestamp,
        # data
        data_sources=data_sources, energy_key=energy_key,
        max_delta_e=max_delta_e,
        physicality_gate=physicality_gate,
        physicality_reject=physicality_reject,
        # training
        batch_size=batch_size,
        num_updates=num_updates, time_limit_hours=time_limit_hours,
        learn_rate=learn_rate, weight_decay=weight_decay, grad_clip=grad_clip,
        duplicate=duplicate, val_duplicate=val_duplicate,
        train_ratio=train_ratio, data_split_seed=data_split_seed,
        sigma_max=sigma_max, eta_window=eta_window,
        use_amp=use_amp, ema_decay=ema_decay,
        stop_check_every=stop_check_every,
        loss_sigma_weighting=loss_sigma_weighting,
        loss_min_snr_gamma=loss_min_snr_gamma,
        log_sigma_bins=log_sigma_bins,
        nan_guard_every=nan_guard_every,
        # model
        irreps_node_x=irreps_node_x, irreps_node_z=irreps_node_z,
        irreps_hidden=irreps_hidden, irreps_edge=irreps_edge, irreps_out=irreps_out,
        num_convs=num_convs, radial_neurons=radial_neurons,
        num_neighbors=num_neighbors, large_cutoff=large_cutoff, cutoff=cutoff,
        n_regions=n_regions,
        conv_type=conv_type, n_radial_basis=n_radial_basis,
        element_latent_dim=element_latent_dim,
        uvu_bottleneck_factor=uvu_bottleneck_factor,
        n_attention_heads=n_attention_heads,
        long_range_cutoff=long_range_cutoff, n_radial_basis_lr=n_radial_basis_lr,
        scalar_cond_names=scalar_cond_names,
        # conditioning v2
        n_methods=n_methods, warm_start_partial=warm_start_partial,
        method_taxonomy=method_taxonomy, delta_e_ref=delta_e_ref,
        cond_ranges=cond_ranges, delta_e_floor=delta_e_floor,
        temperature_mode=temperature_mode, cond_dropout=cond_dropout,
        cond_solo_prob=cond_solo_prob,
        cond_weights=cond_weights,
        rdf_rmax=rdf_rmax, rdf_nbins=rdf_nbins, rdf_channels=rdf_channels,
        provenance=provenance,
        # performance
        num_workers=num_workers, num_proc_workers=num_proc_workers,
        pin_memory=pin_memory,
        # filled in after device / optimizer setup
        device='', amp_dtype='', optimizer_name='AdamW',
    )
