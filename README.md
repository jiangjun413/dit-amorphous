# Data deposit — diffusion-generated amorphous oxides

This repository holds the **primary result artefacts** behind the manuscript:
the trained model checkpoints, the generated configurations, the reference and
calibration energies, the structural-verification output, and the training-loss
curves. Every file is a result. The analysis and generation code that produced
them is not part of this deposit.

Artefacts are identified below by the manuscript's float and section *labels*
rather than by number, because numbering shifts between revisions.

## `checkpoints/`

| file | role | parameters | unpickles with |
|---|---|---|---|
| `production_generator_g2_min1_full_ema.pt` | the generator behind every configuration in `sec:rdf`, `sec:assistedopt`, `sec:versatility` | 595,376 | `dit2` |
| `sparse_corpus_g6_lite_840k_ema.pt` | the 1,781-configuration data-requirement test (`tab:liteverify`) | 606,584 | `dit6` |
| `checkpoint_run_4040k_ema.pt` | final checkpoint of the single run followed in `tab:checkpoints` | 625,669 | `dit6` |

Frozen EMA weights; they load on CPU without CUDA. Parameter counts are
measured from the files, not declared: they differ because the species
embedding scales with the number of chemical species in each run's corpus at
otherwise identical architecture.

Each file is a **pickled model object**, not a bare `state_dict`, so
`torch.load` needs the package that wrote it on `sys.path`. That package is not
in this deposit; `torch.load` will raise `ModuleNotFoundError` without it. The
weights are deposited so that the models behind the reported results are
archived and checkable against `MANIFEST.tsv`. Request the loader from the
corresponding author.

## `structures/`

VASP POSCAR format, all as generated (not relaxed) except the record.

| file | role |
|---|---|
| `record_a-ZrTaO_Ta40Zr40O180_CONTCAR.vasp` | the record amorphous structure of `sec:explore`, −9.57126 eV/atom, DFT-relaxed at fixed cell, ρ = 6.500 g cm⁻³ |
| `generated_GeO2_3000atom.vasp` | the a-GeO₂ partial RDFs of `sec:rdf` |
| `generated_TiGeO_1008atom.vasp` | the six Ti–Ge–O partial RDFs of `sec:rdf` |
| `generated_interface_SiO2_TiGeO2_480atom.vasp` | the interface of `fig:interface` |
| `generated_multilayer_4layer_36000atom.vasp` | the four-layer, three-oxide stack of `fig:interface` |
| `generated_bilayer_138345atom.vasp` | the largest multilayer of `fig:interface` |
| `generated_HfO2_27pct_SiO2_9999atom.vasp` | the mixed Hf–Si–O cell of `sec:versatility` |

## `energies/`

- `ENERGY_REFERENCES.md` — the per-family relaxed-crystal energy and
  lowest-known amorphous energy that define the two references of
  `sec:assistedopt`. Excess energy is measured against these throughout.
- `conditioning_calibration_scan6.csv` — 1,428 rows, the per-configuration DFT
  energies behind the conditioning calibration of `sec:conditioning` and
  `fig:geo2calib`: requested and achieved excess energy, density, checkpoint,
  minimum interatomic distance and structural-validity flag for every cell.
- `density_energy_map.txt` — 193,206 configurations as
  `density, energy/atom, atom count, per-element counts`; the (ρ, E) coverage
  map of `sec:versatility`.

## `training_loss/`

One CSV per backbone — `graphite.csv`, `uvu.csv`, `dual_cutoff.csv` — each 431
epochs of `epoch, train_mse_online, train_mse_eval, valid_mse, valid_mse_ema,
epoch_seconds`, extracted from the run logs of the three-backbone comparison.
These are the curves plotted in the training-loss figure. Best validation loss
is 0.0743 at epoch 281 (graphite), 0.0794 at epoch 361 (UVU) and 0.0675 at
epoch 423 (dual-cutoff), which is how these three runs were identified among
the archived logs.

## `verification/lite_verify_20260819/`

The structural-verification campaign for the sparse-corpus model
(`tab:liteverify`): partial RDFs and coordination-number histograms for
generated cells and their DFT references, computed through one code path for
both sides. `analysis/*.npz` holds the raw g(r) arrays and CN histograms;
`results.csv` is the machine-readable summary; `tables.md` and `RESULTS.md` are
the campaign's own write-ups; `refs.json` records which DFT calculation each
reference came from.

## Provenance and integrity

`MANIFEST.tsv` lists every deposited file with its size and SHA-256. Absolute
source paths, account names and compute-node names have been removed from the
provenance fields; reference paths in `refs.json` and `results.csv` are
rewritten relative to `<dft-archive>/`, keeping the identity of each
calculation without the archive's internal layout.

## What is not here

- **The code.** Training, generation and analysis code is not deposited here.
- **The coordinates of the N = 300,000-atom a-GeO₂ cell.** Not retained after
  the run; only the render survives. That run is quoted for generation cost and
  system size, and the structural comparison is made at the deposited cell size.
- **The raw DFT electronic-structure output** the training set was derived from
  (92,721 `vasprun.xml` decks, well over 100 GB). Available from the
  corresponding author on reasonable request.
- **The derived training and validation trajectories.** Not in this deposit.
