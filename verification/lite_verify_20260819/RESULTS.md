# g6_lite structural verification — partial RDFs and coordination numbers
### four amorphous oxides vs DFT references, 2026-08-19

**Model.** `geo2_wide_20260819/g6_lite_840k_ema_frozen.pt` — dit6 package, graphite
trunk, **606,584 parameters**, frozen EMA weights at step 840,000. Trained on the
**lite corpus: 1,781 configurations drawn from 430 DFT trajectories over 20
chemistries** (measured, not quoted: `07_trainingset_lite/SELECTION.csv` has 430
rows; the training logs report `Total: 1781 | Train: 1602 | Valid: 179`; element
alphabet as reported by the generator is `O, Si, Ar, Ti, Fe, Ge, Zr, Nb, Hf, Ta`).
Note: the brief said 441 trajectories; **the measured number is 430** and no
artifact on disk contains 441.

**Generation.** 22 cells, all at `target_delta_e_min = 0.000` with
`target_energy` = the checkpoint-era ΔE floor from `docs/ENERGY_REFERENCES.md`
(GeO2 −6.4138, TiGeO −7.5590, TiO2 −8.9460, HfO2 −11.0900 eV/atom),
`refine_sigma = 0.02`, `force_relax_steps = 0`. GPUs 1 and 2 of `<compute-node>`
(L4); 74–99 s per 300–360-atom cell, 769 s for the 3000-atom cell. No SLURM job
was submitted and nothing was installed into `dm2`.

**Analysis.** `rdf_cn.py` (one code path for generated cells and DFT frames):
partial g(r) from an `ase` neighbour list with periodic images (NOT minimum
image — a-HfO2 at ρ 9.4 is a 15.5 Å box and L/2 < r_max), dr = 0.02 Å, r_max
8 Å; first-peak position and FWHM read off a 0.10 Å boxcar; CN from the campaign
first-shell cutoffs **Ge–O 2.40, Ti–O 2.60, Hf–O 2.90 Å**
(`summary/analyze_gen_tests.py`, not invented here); min interatomic distance
against the project's 1.45 Å chemistry gate.

**The analysis code is validated, not asserted** (`validate_rdf_code.py`):
1. Against a *different* implementation — the campaign's own 1000-frame average
   of the same trajectory (`07_trainingset/08_GeO/300K_MD/vasprun_partial_rdf.txt`,
   produced years earlier by `00_others/rdf_ave_gemini.py` using MIC and
   dr = 0.05 Å). First-peak positions agree to within one bin:
   Ge–Ge 3.129 vs 3.170, Ge–O 1.7772 vs 1.770, O–O 2.8785 vs 2.890 Å.
2. Against a completely separate code path — integrating
   4πr²ρ_b g_ab(r) dr to the bond cutoff reproduces the directly counted CN
   **exactly**: CN(Ge→O) 4.0425 vs 4.0425, CN(O→Ge) 2.0212 vs 2.0212. The
   campaign file integrates to 4.0424 on the same footing.

---

## Headline table (density-matched, 0 K DFT inherent-structure references)

The primary comparison is against **DFT ionic relaxations at the same density**,
because the generator is run at T_eff = 0 and returns an inherent structure;
comparing its peak *widths* against a thermally broadened MD average would be a
category error. Finite-T references are reported separately below.

| system | gen atoms | ρ (g/cc) gen / DFT | reference | ref atoms / frames | ref **in lite corpus?** | first-peak r₁ gen vs DFT (Å) | Δr₁ (Å) | mean CN gen vs DFT | Δ CN | d_min gen (Å) | gate ≥1.45 Å |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **a-GeO₂** | 360 (×3) | 3.746 / 3.746 | `GeO/relax/s00_08_GeO_relax_300K` (last 10 ionic steps) | 360 / 10 | **IN** | Ge–O 1.770 vs 1.770 · O–O 2.877 vs 2.890 · Ge–Ge 3.203 vs 3.150 | +0.000 · −0.013 · +0.053 | Ge 4.050±0.022 vs 4.042 · O 2.025 vs 2.021 | +0.008 · +0.004 | 1.707±0.012 | PASS |
| **a-Ti₀.₄₄Ge₀.₅₆O₂** | 300 (×3) | 3.700 / 3.700 | `TiGeO/relax/s00_07_TiGeO_01_relax_002_720_44_0_3.7_…_01_DFT_relax` (last 10) | **720** / 10 | **NOT in** | Ti–O 1.837 vs 1.830 · Ge–O 1.770 vs 1.770 · O–O 2.883 vs 2.870 · Ti–Ge 3.450 vs 3.510 · Ge–Ge 3.223 vs 3.190 · Ti–Ti 3.577 vs 3.450 | +0.007 · +0.000 · +0.013 · −0.060 · +0.033 · +0.127 | Ti 5.045±0.023 vs 5.089 · Ge 4.107±0.036 vs 4.037 · O 2.260 vs 2.248 | −0.043 · +0.070 · +0.012 | 1.693±0.007 | PASS |
| **a-TiO₂** | 300 (×3) | 2.660 / 2.660 | `TiO/relax/s10_relax11_r11_TiO_sb18_r28k0x0` (last 10) | 240 / 10 | **NOT in** | Ti–O 1.830 vs 1.830 · O–O 2.950 vs 2.950 · Ti–Ti 3.590 vs 3.630 | +0.000 · +0.000 · −0.040 | Ti 4.607±0.045 vs 4.675 · O 2.303 vs 2.337 | −0.068 · −0.034 | 1.739±0.009 | PASS |
| **a-HfO₂** | 300 (×3) | 9.400 / 9.400 | `HfO/relax/s10_relax11_r11_HfO_sb17_r11k0x0` (last 10) | 240 / 10 | **NOT in** | Hf–O 2.110 vs 2.090 · O–O 2.650 vs 2.610 · Hf–Hf 3.343 vs 3.450 | +0.020 · +0.040 · −0.107 | Hf 6.620±0.060 vs 6.670 · O 3.310 vs 3.335 | −0.050 · −0.025 | 1.902±0.014 | PASS |

**Every cation–oxygen first-peak position is reproduced to ≤0.02 Å and every
mean CN to ≤0.07** across all four chemistries, three of the four against a DFT
reference that is **not** in the 1,781-configuration training set.

## Secondary pairings

| system | gen atoms | ρ gen / DFT | reference | in lite? | worst Δr₁ (Å) | Δ mean CN (cation) | verdict |
|---|---|---|---|---|---|---|---|
| a-GeO₂ | 360 (×3) | 3.650 / 3.683 | `GeO/relax/s04_vasp_relax_comp_00_001` (0 K, last 10) | **NOT in** | Ge–Ge −0.080 (Ge–O +0.000) | Ge +0.033 | reproduces |
| a-TiO₂ | 300 (×3) | 2.700 / 2.680 | `TiO/relax/s10_relax13_r13_TiO_sb18_r16k1x0` (0 K, last 10; the campaign AMF, −8.8548 eV/atom) | IN | Ti–Ti +0.027 (Ti–O +0.000) | Ti −0.038 | reproduces |
| a-GeO₂ | 360 (×3) | 3.746 / 3.746 | `GeO/md_nvt/s00_08_GeO_300K_MD` — **the reference behind paper Fig. 8** | IN | Ge–Ge +0.033 (Ge–O +0.000) | Ge +0.007 | reproduces |
| a-HfO₂ | 300 (×3) | 8.897 / 8.900 | `HfO/relax/s00_16_HfO_03_best_aHfO2_03_md_relax_s{01,02,04,05}` (0 K, last 5 each = 20 frames) | **NOT in** | Hf–Hf +0.220, O–O +0.107 (Hf–O +0.007) | **Hf −0.438** | **FAILS on CN** |
| a-HfO₂ | 300 (×3) | 8.897 / 8.900 | `HfO/md_nvt/s00_16_HfO_03_best_aHfO2_02_md_300K` (300 K Langevin, every 3rd of 3000 frames) | IN | Hf–Hf +0.220, O–O +0.107 (Hf–O +0.007) | **Hf −0.433** | **FAILS on CN** |

## Size independence — a-GeO₂ at 360 vs 3000 atoms

Same yaml, same density (3.650), same conditioning, one 3000-atom cell
(Ge₁₀₀₀O₂₀₀₀, 36.24 Å box, 769 s on one L4):

| cell | mean CN(Ge) | mean CN(O) | d_min (Å) | Ge–O r₁ / FWHM (Å) | O–O r₁ (Å) | Ge–Ge r₁ (Å) |
|---|---|---|---|---|---|---|
| 360 atoms (3 repeats) | 4.053 ± 0.010 | 2.026 ± 0.005 | 1.695 ± 0.007 | 1.770 / 0.103 ± 0.001 | 2.870 ± 0.020 | 3.190 ± 0.000 |
| 3000 atoms (1 cell) | 4.035 | 2.018 | 1.669 | 1.770 / 0.102 | 2.890 | 3.250 |

The only quantity that moves by more than the 360-atom repeat spread is the
medium-range Ge–Ge peak (3.250 vs 3.190 Å); short-range structure, coordination
and the validity gate are size-independent.

All three partials overlay within the counting noise of the 360-atom cells
(`figures/fig_size_independence.pdf`).

---

## Per-system verdicts

**a-GeO₂ — REPRODUCED.** Ge–O 1.770 Å against 1.770 Å (0 K relaxation) and
1.770 Å (1000-frame MD): exact to the 0.02 Å bin. O–O within 0.013 Å, Ge–Ge
within 0.053 Å. Mean CN(Ge) 4.050 ± 0.022 vs 4.042; 95.0 % vs 95.8 % four-fold.
Ge–O FWHM 0.102 vs 0.101 Å — the width, not just the position, matches the 0 K
reference. Holds at both densities tested and at 3000 atoms.

**a-Ti₀.₄₄Ge₀.₅₆O₂ — REPRODUCED, and this is the strongest single result.** The
reference is the manuscript's own Fig. 9 object: a **720-atom** Ti₁₀₅Ge₁₃₅O₄₈₀
(43.75 % Ti) DFT relaxation at ρ 3.701, and it is **not in the lite corpus**.
All six partials agree: both cation–O peaks to ≤0.007 Å, O–O to 0.013 Å, and
even Ti–Ti — the rarest pair, 44 Ti atoms in a 300-atom cell — to 0.127 Å with a
per-repeat spread of ±0.050 Å. Mean CN(Ti) 5.045 vs 5.089, CN(Ge) 4.107 vs
4.037. The one visible deficiency is the Ge CN₄ fraction, 89.9 % vs 96.3 %: the
model puts ~6 % more Ge in five-fold sites than DFT does.

**a-TiO₂ — REPRODUCED.** Ti–O 1.830 Å exactly, O–O 2.950 Å exactly, Ti–Ti within
0.040 Å; mean CN(Ti) 4.607 ± 0.045 vs 4.675, against an out-of-corpus reference
at exactly matched density. The low mean CN(Ti) ≈ 4.6 (rather than the ~6 of
crystalline TiO₂) is a genuine property of this low-density amorphous basin, not
a model artifact — **the DFT reference gives the same 4.675**. Also reproduced at
ρ 2.700 against the in-corpus AMF structure.

**a-HfO₂ — REPRODUCED AT ρ 9.400, FAILS AT ρ 8.897.** At the canonical density
the agreement is good against an out-of-corpus reference: Hf–O 2.110 vs
2.090 Å, mean CN(Hf) 6.620 ± 0.060 vs 6.670, CN₇ 50.7 % vs 60.8 %. The Hf–Hf
medium-range peak is consistently short, 3.343 vs 3.450 Å (−0.107 Å, spread
±0.012 Å — reproducible, not noise). **At ρ 8.897 the model under-coordinates
badly: mean CN(Hf) 6.323 ± 0.031 against 6.761 for the 0 K inherent structures
and 6.756 for the 300 K MD, a deficit of −0.44.** This is not a thermal artifact:
the 0 K and 300 K references agree with each other to 0.005 and both disagree
with the generated cells. It is not a density-mismatch artifact either — gen and
ref are at 8.897 vs 8.900. The generated CN(Hf) tracks density (6.62 at 9.40 →
6.32 at 8.90) whereas DFT a-HfO₂ holds CN(Hf) ≈ 6.7–6.8 across the same range,
so **g6_lite's a-HfO₂ coordination is too density-sensitive.** Caveat on the
reference's own statistics: it is a 96-atom cell (32 Hf), so its mean CN carries
roughly ±0.06; the −0.44 gap is far outside that.

## Limitations that must be stated in the paper

1. **Reference provenance for a-TiO₂ and a-HfO₂ is partly circular.** The
   original corpus contains **no** amorphous TiO₂ DFT data at all
   (`07_trainingset/09_TiO/` is ten Materials-Project *crystal* polymorphs, 6–48
   atoms, all IBRION=2 EOS relaxations) and **no** a-TiO₂ or a-Ti-Ge-O DFT-MD
   anywhere. Every multi-hundred-atom a-TiO₂ DFT object in the repo is a DFT
   relaxation whose starting configuration came from a diffusion model. The
   relaxation itself is genuine DFT and moves the atoms substantially, so the
   endpoint is a true DFT inherent structure — but the *basin* was chosen by a
   model, and that must not be presented as a fully independent test. a-GeO₂ and
   a-HfO₂ are clean on this point (melt-quench / MD origin).
2. **Only a-GeO₂ and a-HfO₂ have finite-T DFT-MD at all.** a-TiO₂ and
   a-Ti-Ge-O have none in this repo.
3. **The a-GeO₂ MD reference is a temperature ramp, not isothermal 300 K.**
   `TEBEG = 10 → TEEND = 300 K`, `SMASS = 0`, 1000 × 1 fs. Temperatures computed
   from the per-frame kinetic energy: mean **155.1 K**, median 170.5 K, max
   322.5 K, mean of the last 200 frames 270.2 K. The manuscript caption
   "1000-frame average of a 360-atom DFT-MD trajectory at 300 K" is right about
   the frame count and cell size but glosses the ramp. The a-HfO₂ MD *is*
   isothermal (`MDALGO = 3` Langevin, `TEBEG = TEEND = 300`, 3000 × 1 fs; mean
   292.0 K, median 297.3 K).
4. **The TiGeO reference is a relaxation, not MD.** All 377 vasprun.xml under
   `07_trainingset/07_TiGeO/` have IBRION = 2; the `…1000ps` in the directory
   names is the *classical* pre-quench time. The manuscript's phrase "roughly two
   weeks of cluster time for the 720-atom DFT-MD trajectory" describes DFT
   relaxations of classically quenched cells.
5. **Peak positions are bin-limited to ±0.02 Å**, and medium-range
   cation–cation peaks carry a real repeat-to-repeat spread of ±0.05–0.23 Å at
   these cell sizes. Quote the cation–O numbers as the sharp result.
6. **n = 3 per system.** Spreads are sample s.d. over three cells, not
   converged error bars.

## Files

| path | contents |
|---|---|
| `gen/*.yaml` | 22 generation configs (4 chemistries × 3 repeats at canonical ρ; 3 chemistries × 3 at DFT-matched ρ; 1 × 3000-atom) |
| `queue.txt`, `queue2.txt`, `run_pool.sh`, `run_pool2.sh` | pooled two-GPU generation drivers |
| `out/<tag>/` | generated structures (`*_final.vasp`), generator RDF/CN sidecars, plots |
| `rdf_cn.py` | the analysis library (partial RDFs, first-peak metrics, CN, d_min) |
| `run_analysis.py` | driver: reads `refs.json`, caches to `analysis/*.npz`, writes `results.csv` |
| `refs.json` | the reference declaration — paths, frame selection, lite-corpus flags, provenance notes |
| `validate_rdf_code.py` | the two independent validations of the RDF code |
| `make_tables.py`, `tables.md` | tables generated from `results.csv` (nothing retyped) |
| `plot_figures.py`, `figures/` | four figures, PDF + PNG at 300 dpi |
| `results.csv` | 31 rows — every analysed object, every number |

Rerun end to end with:
```
bash run_pool.sh g1 1 &  bash run_pool2.sh h2 2 &     # generation
python run_analysis.py && python validate_rdf_code.py
python make_tables.py > tables.md && python plot_figures.py
```

---

### Per-pairing summary

| system | gen atoms | gen rho | reference | ref atoms | ref frames | ref rho | ref in lite corpus? | pair | r1 gen (A) | r1 DFT (A) | delta r1 (A) | FWHM gen (A) | FWHM DFT (A) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GeO2m | 360 | 3.746 | GeO2_relax_0K | 360 | 10 | 3.7459 | True | Ge-Ge | 3.203 ± 0.046 | 3.150 | +0.053 | 0.298 ± 0.011 | 0.292 |
|  |  |  |  |  |  |  |  | Ge-O | 1.770 ± 0.000 | 1.770 | +0.000 | 0.102 ± 0.001 | 0.101 |
|  |  |  |  |  |  |  |  | O-O | 2.877 ± 0.012 | 2.890 | -0.013 | 0.239 ± 0.014 | 0.278 |
| TiGeO | 300 | 3.7 | TiGeO_720atom_0K | 720 | 10 | 3.6998 | False | Ge-Ge | 3.223 ± 0.081 | 3.190 | +0.033 | 0.230 ± 0.055 | 0.189 |
|  |  |  |  |  |  |  |  | Ge-O | 1.770 ± 0.000 | 1.770 | +0.000 | 0.107 ± 0.001 | 0.104 |
|  |  |  |  |  |  |  |  | O-O | 2.883 ± 0.023 | 2.870 | +0.013 | 0.336 ± 0.031 | 0.321 |
|  |  |  |  |  |  |  |  | Ti-Ti | 3.577 ± 0.050 | 3.450 | +0.127 | 0.276 ± 0.090 | 0.415 |
|  |  |  |  |  |  |  |  | Ti-Ge | 3.450 ± 0.035 | 3.510 | -0.060 | 0.345 ± 0.038 | 0.312 |
|  |  |  |  |  |  |  |  | Ti-O | 1.837 ± 0.012 | 1.830 | +0.007 | 0.142 ± 0.023 | 0.144 |
| TiO2m | 300 | 2.66 | TiO2_relax11_0K | 240 | 10 | 2.66 | False | O-O | 2.950 ± 0.020 | 2.950 | +0.000 | 0.337 ± 0.039 | 0.368 |
|  |  |  |  |  |  |  |  | Ti-Ti | 3.590 ± 0.000 | 3.630 | -0.040 | 0.338 ± 0.029 | 0.344 |
|  |  |  |  |  |  |  |  | Ti-O | 1.830 ± 0.000 | 1.830 | +0.000 | 0.119 ± 0.007 | 0.127 |
| HfO2 | 300 | 9.4 | HfO2_relax11_0K | 240 | 10 | 9.4 | False | O-O | 2.650 ± 0.053 | 2.610 | +0.040 | 0.639 ± 0.062 | 0.616 |
|  |  |  |  |  |  |  |  | Hf-Hf | 3.343 ± 0.012 | 3.450 | -0.107 | 0.443 ± 0.099 | 0.340 |
|  |  |  |  |  |  |  |  | Hf-O | 2.110 ± 0.020 | 2.090 | +0.020 | 0.197 ± 0.012 | 0.196 |
| GeO2 | 360 | 3.65 | GeO2_comp00_0K | 300 | 10 | 3.683 | False | Ge-Ge | 3.190 ± 0.000 | 3.270 | -0.080 | 0.306 ± 0.022 | 0.329 |
|  |  |  |  |  |  |  |  | Ge-O | 1.770 ± 0.000 | 1.770 | +0.000 | 0.103 ± 0.001 | 0.101 |
|  |  |  |  |  |  |  |  | O-O | 2.870 ± 0.020 | 2.870 | -0.000 | 0.227 ± 0.007 | 0.279 |
| TiO2 | 300 | 2.7 | TiO2_relax13_0K | 240 | 10 | 2.68 | True | O-O | 2.930 ± 0.020 | 2.930 | +0.000 | 0.349 ± 0.012 | 0.321 |
|  |  |  |  |  |  |  |  | Ti-Ti | 3.557 ± 0.042 | 3.530 | +0.027 | 0.276 ± 0.016 | 0.351 |
|  |  |  |  |  |  |  |  | Ti-O | 1.830 ± 0.000 | 1.830 | +0.000 | 0.126 ± 0.009 | 0.113 |
| GeO2m | 360 | 3.746 | GeO2_MD_10to300K | 360 | 1000 | 3.7459 | True | Ge-Ge | 3.203 ± 0.046 | 3.170 | +0.033 | 0.298 ± 0.011 | 0.348 |
|  |  |  |  |  |  |  |  | Ge-O | 1.770 ± 0.000 | 1.770 | +0.000 | 0.102 ± 0.001 | 0.102 |
|  |  |  |  |  |  |  |  | O-O | 2.877 ± 0.012 | 2.890 | -0.013 | 0.239 ± 0.014 | 0.325 |
| HfO2m | 300 | 8.897 | HfO2_inherent_0K | 96 | 20 | 8.9 | False | O-O | 2.737 ± 0.103 | 2.630 | +0.107 | 0.727 ± 0.032 | 0.344 |
|  |  |  |  |  |  |  |  | Hf-Hf | 3.630 ± 0.225 | 3.450 | +0.180 | 0.708 ± 0.292 | 0.240 |
|  |  |  |  |  |  |  |  | Hf-O | 2.097 ± 0.012 | 2.090 | +0.007 | 0.197 ± 0.030 | 0.173 |
| HfO2m | 300 | 8.897 | HfO2_MD_300K | 96 | 1000 | 8.9 | True | O-O | 2.737 ± 0.103 | 2.630 | +0.107 | 0.727 ± 0.032 | 0.630 |
|  |  |  |  |  |  |  |  | Hf-Hf | 3.630 ± 0.225 | 3.410 | +0.220 | 0.708 ± 0.292 | 0.501 |
|  |  |  |  |  |  |  |  | Hf-O | 2.097 ± 0.012 | 2.090 | +0.007 | 0.197 ± 0.030 | 0.252 |

### Coordination numbers and validity gate

| system | reference | element | mean CN gen | mean CN DFT | delta CN | CN4 gen / DFT | CN6 gen / DFT | CN7 gen / DFT | dmin gen (A) | dmin DFT (A) | dmin >= 1.45 A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GeO2m | GeO2_relax_0K | Ge | 4.050 ± 0.022 | 4.042 | +0.008 | 0.950 / 0.958 | 0.000 / 0.000 | 0.000 / 0.000 | 1.707 ± 0.012 | 1.728 | PASS |
|  |  | O | 2.025 ± 0.011 | 2.021 | +0.004 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |  |  |  |
| TiGeO | TiGeO_720atom_0K | Ti | 5.045 ± 0.023 | 5.089 | -0.043 | 0.205 / 0.152 | 0.250 / 0.241 | 0.000 / 0.000 | 1.693 ± 0.007 | 1.715 | PASS |
|  |  | Ge | 4.107 ± 0.036 | 4.037 | +0.070 | 0.899 / 0.963 | 0.006 / 0.000 | 0.000 / 0.000 |  |  |  |
|  |  | O | 2.260 ± 0.015 | 2.248 | +0.012 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |  |  |  |
| TiO2m | TiO2_relax11_0K | Ti | 4.607 ± 0.045 | 4.675 | -0.068 | 0.467 / 0.412 | 0.073 / 0.087 | 0.000 / 0.000 | 1.739 ± 0.009 | 1.738 | PASS |
|  |  | O | 2.303 ± 0.023 | 2.337 | -0.034 | 0.010 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |  |  |  |
| HfO2 | HfO2_relax11_0K | Hf | 6.620 ± 0.060 | 6.670 | -0.050 | 0.000 / 0.000 | 0.397 / 0.343 | 0.507 / 0.608 | 1.902 ± 0.014 | 1.906 | PASS |
|  |  | O | 3.310 ± 0.030 | 3.335 | -0.025 | 0.315 / 0.341 | 0.000 / 0.000 | 0.000 / 0.000 |  |  |  |
| GeO2 | GeO2_comp00_0K | Ge | 4.053 ± 0.010 | 4.020 | +0.033 | 0.947 / 0.980 | 0.000 / 0.000 | 0.000 / 0.000 | 1.695 ± 0.007 | 1.717 | PASS |
|  |  | O | 2.026 ± 0.005 | 2.010 | +0.016 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |  |  |  |
| TiO2 | TiO2_relax13_0K | Ti | 4.637 ± 0.042 | 4.675 | -0.038 | 0.460 / 0.450 | 0.097 / 0.125 | 0.000 / 0.000 | 1.728 ± 0.018 | 1.720 | PASS |
|  |  | O | 2.318 ± 0.021 | 2.337 | -0.019 | 0.010 / 0.013 | 0.000 / 0.000 | 0.000 / 0.000 |  |  |  |
| GeO2m | GeO2_MD_10to300K | Ge | 4.050 ± 0.022 | 4.043 | +0.007 | 0.950 / 0.961 | 0.000 / 0.003 | 0.000 / 0.000 | 1.707 ± 0.012 | 1.672 | PASS |
|  |  | O | 2.025 ± 0.011 | 2.021 | +0.004 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |  |  |  |
| HfO2m | HfO2_inherent_0K | Hf | 6.323 ± 0.031 | 6.761 | -0.438 | 0.000 / 0.000 | 0.593 / 0.292 | 0.320 / 0.406 | 1.906 ± 0.002 | 1.861 | PASS |
|  |  | O | 3.162 ± 0.015 | 3.381 | -0.219 | 0.210 / 0.459 | 0.000 / 0.000 | 0.000 / 0.000 |  |  |  |
| HfO2m | HfO2_MD_300K | Hf | 6.323 ± 0.031 | 6.756 | -0.433 | 0.000 / 0.001 | 0.593 / 0.331 | 0.320 / 0.483 | 1.906 ± 0.002 | 1.794 | PASS |
|  |  | O | 3.162 ± 0.015 | 3.378 | -0.217 | 0.210 / 0.445 | 0.000 / 0.006 | 0.000 / 0.000 |  |  |  |

### Every analysed object (raw)

| system | kind | label | atoms | frames | rho (g/cc) | dmin (A) | mean CN (per element) |
|---|---|---|---|---|---|---|---|
| GeO2m | generated | GeO2m_r0 | 360 | 1 | 3.746 | 1.6943 | Ge 4.042, O 2.021 |
| GeO2m | generated | GeO2m_r1 | 360 | 1 | 3.746 | 1.7189 | Ge 4.033, O 2.017 |
| GeO2m | generated | GeO2m_r2 | 360 | 1 | 3.746 | 1.7086 | Ge 4.075, O 2.038 |
| GeO2m | DFT | GeO2_relax_0K | 360 | 10 | 3.7459 | 1.7284 | Ge 4.042, O 2.021 |
| GeO2m | DFT | GeO2_MD_10to300K | 360 | 1000 | 3.7459 | 1.6722 | Ge 4.043, O 2.021 |
| GeO2 | generated | GeO2_r0 | 360 | 1 | 3.65 | 1.6886 | Ge 4.058, O 2.029 |
| GeO2 | generated | GeO2_r1 | 360 | 1 | 3.65 | 1.6952 | Ge 4.058, O 2.029 |
| GeO2 | generated | GeO2_r2 | 360 | 1 | 3.65 | 1.7017 | Ge 4.042, O 2.021 |
| GeO2 | DFT | GeO2_comp00_0K | 300 | 10 | 3.683 | 1.7173 | Ge 4.020, O 2.010 |
| GeO2big | generated | GeO2big_r0 | 3000 | 1 | 3.65 | 1.6686 | Ge 4.035, O 2.018 |
| TiGeO | generated | TiGeO_r0 | 300 | 1 | 3.7 | 1.6998 | Ge 4.107, O 2.260, Ti 5.045 |
| TiGeO | generated | TiGeO_r1 | 300 | 1 | 3.7 | 1.6927 | Ge 4.143, O 2.275, Ti 5.068 |
| TiGeO | generated | TiGeO_r2 | 300 | 1 | 3.7 | 1.6855 | Ge 4.071, O 2.245, Ti 5.023 |
| TiGeO | DFT | TiGeO_720atom_0K | 720 | 10 | 3.6998 | 1.715 | Ge 4.037, O 2.248, Ti 5.089 |
| TiO2m | generated | TiO2m_r0 | 300 | 1 | 2.66 | 1.7489 | O 2.280, Ti 4.560 |
| TiO2m | generated | TiO2m_r1 | 300 | 1 | 2.66 | 1.7376 | O 2.305, Ti 4.610 |
| TiO2m | generated | TiO2m_r2 | 300 | 1 | 2.66 | 1.732 | O 2.325, Ti 4.650 |
| TiO2m | DFT | TiO2_relax11_0K | 240 | 10 | 2.66 | 1.7376 | O 2.337, Ti 4.675 |
| TiO2 | generated | TiO2_r0 | 300 | 1 | 2.7 | 1.7461 | O 2.295, Ti 4.590 |
| TiO2 | generated | TiO2_r1 | 300 | 1 | 2.7 | 1.7103 | O 2.335, Ti 4.670 |
| TiO2 | generated | TiO2_r2 | 300 | 1 | 2.7 | 1.7264 | O 2.325, Ti 4.650 |
| TiO2 | DFT | TiO2_relax13_0K | 240 | 10 | 2.68 | 1.7204 | O 2.337, Ti 4.675 |
| HfO2 | generated | HfO2_r0 | 300 | 1 | 9.4 | 1.9099 | O 3.280, Hf 6.560 |
| HfO2 | generated | HfO2_r1 | 300 | 1 | 9.4 | 1.8853 | O 3.310, Hf 6.620 |
| HfO2 | generated | HfO2_r2 | 300 | 1 | 9.4 | 1.9104 | O 3.340, Hf 6.680 |
| HfO2 | DFT | HfO2_relax11_0K | 240 | 10 | 9.4 | 1.9063 | O 3.335, Hf 6.670 |
| HfO2m | generated | HfO2m_r0 | 300 | 1 | 8.897 | 1.9071 | O 3.175, Hf 6.350 |
| HfO2m | generated | HfO2m_r1 | 300 | 1 | 8.897 | 1.9038 | O 3.145, Hf 6.290 |
| HfO2m | generated | HfO2m_r2 | 300 | 1 | 8.897 | 1.9059 | O 3.165, Hf 6.330 |
| HfO2m | DFT | HfO2_inherent_0K | 96 | 20 | 8.9 | 1.8611 | O 3.381, Hf 6.761 |
| HfO2m | DFT | HfO2_MD_300K | 96 | 1000 | 8.9 | 1.7935 | O 3.378, Hf 6.756 |
