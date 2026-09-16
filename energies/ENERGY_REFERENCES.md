# Canonical energy references (eV/atom, VASP-PBE, campaign protocol)

Updated 2026-07-29 after the crystal-reference campaign (`xtl_ref_20260729/`:
Γ-only 216–384-atom supercells, ISIF=3 relax @ ENCUT 520 → fixed-cell SP with
INCAR_sp, same POTCARs as all amorphous SPs). Three tiers — do not mix them:

## 1. E_XTL — physical crystal reference (true thermodynamic ΔE=0)

| family | E_XTL | reference structure |
|---|---|---|
| GeO₂ | −6.4141 | α-quartz −6.4133 ≈ amorphous −6.4141 (PBE-degenerate); rutile −6.4083 (PBE artifact) |
| SiO₂ | **−7.9045** | α-quartz (COD 5000035) — 40 meV below the glass |
| Si | −5.424 | diamond (corpus) |
| TiO₂ | −8.946 | anatase (corpus; our Γ-only anatase −8.9398 confirms within 6 meV; rutile −8.9095) |
| HfO₂ | −11.090 | monoclinic (corpus) |
| NbO₂ | −9.2824 | crystal (corpus) |
| TiGeO (Ti₄₄Ge₅₆O₂₀₀) | −7.559 | corpus structure — BELOW the anatase+quartz tie-line (−7.525): favorable mixing |
| Zr-Ta-O (Ta₄₀Zr₄₀O₁₈₀) | **−9.6795** | tie-line 40 m-ZrO₂ (−9.5936) + 20 B-Ta₂O₅ (−9.7532); β-Ta₂O₅ (−9.4339) rejected — frustrated Pccm model |

## 2. AMF — best known amorphous structure (generation saturation floor)

| family | AMF | amorphization energy E_am − E_XTL |
|---|---|---|
| GeO₂ | −6.4141 | ~0 (glass ≈ quartz at PBE) |
| SiO₂ | −7.864 | +0.041 |
| Si | −5.267 | +0.157 |
| TiO₂ | **−8.85477** (relax13 sb18 r16k1x0, converged fmax 0.036, 2026-08-02; prev −8.8542 relax11 sb18 r28k0 ρ2.66; basin flat 2.6–2.9) | +0.092 |
| HfO₂ | **−10.93669** (relax13 sb17 r11k3x0, converged fmax 0.043, 2026-08-02; runner-up −10.93626 sb13 r13k2x0; prev −10.9348 relax11 sb17 r11k0 cont. ρ9.40) | +0.157 |
| NbO₂ | −9.1401 | +0.142 |
| Zr-Ta-O | **−9.57126** (relax13 sb9 r2k0x2, converged fmax 0.035, 2026-08-02; prev −9.5703 relax9 sb6 r0x4 ρ6.42 — reproduced exactly at −9.57028) | +0.110 |
| ZrO₂ | **−9.45167** (2026-08-31; `07_trainingset/13_ZrO/02_amorphous/01_v1`, 876 atoms) | +0.142 |
| Ta₂O₅ | **−9.70360** (2026-08-31; `07_trainingset/10_TaO/02_amorphous/02_v2`, 1008 atoms) | +0.050 |

The last two rows were added 2026-08-31 because multiscan11 scans **ZrO₂ and
Ta₂O₅ as separate systems**, and only the mixed Zr-Ta-O phase had an AMF — so
both of those panels were being scored against a crystal with no amorphous
reference at all. They were derived by `multiscan11_20260829/detect_amf.py`,
which sweeps all 46 corpus pickles for structures matching the scan cell's
composition signature and takes the lowest-energy amorphous one.

**That script reproduces every pre-existing AMF row above exactly** (TiO₂
−8.85477, HfO₂ −10.93669, SiO₂ −7.864, Si −5.267, NbO₂ −9.1401, GeO₂ −6.4141),
which is why the two new numbers are trusted. Caveat on their weight: ZrO₂ rests
on 36 amorphous structures and Ta₂O₅ on 18, both from a single structure family,
against 1,233–2,283 for the well-sampled systems. Ta₂O₅'s +0.050 also breaks the
"+0.10…+0.16 uniform" pattern noted below, so it is the one most likely to move
if a proper relax campaign is run on it.

**Classification must be by PATH, not by the `phases` label.** Two traps found
2026-08-31: 36 heated-crystal Si frames (`xtl__Si1__T225`) carry
`phase='unknown'` and sit 110 meV BELOW the true a-Si minimum; and a 12-atom
`mp-554278` cell was picked up as a-TiO₂ at exactly E_XTL, which would have
erased the entire 91 meV TiO₂ correction. A Materials-Project id in the path
means crystal UNLESS the path also shows an amorphising step (`melt`, `hotmd`,
`quench`, …) — `08_GeO/melt/mp7812` is genuinely amorphous.

Amorphization energies are now uniform (+0.10…+0.16) across all oxides once the
proper crystal references are used — a strong consistency check.

**K-point validation (xtl_ref_20260729/ktest, 2×2×2 Γ-centered vasp_std on the
relaxed supercells):** every adopted reference is Γ-converged — SiO₂-qz 0.02,
ZrO₂-m 0.09, GeO₂-qz 0.16, B-Ta₂O₅ 0.40, TiO₂-rt/an 0.83/0.97 meV/atom
(GeO₂-rt 3.7, still passing). Sole failure: β-Ta₂O₅ at 8.85 meV/atom — the
already-rejected frustrated polymorph (k-converged it is still +0.31 above
B-phase). No table changes required.

## 3. E_TRAIN — floors baked into each model's ΔE conditioning labels

Generation conditioning MUST use the training-era floor of the checkpoint in use
(see memory: energy-conditioning-scale-drift). For g2_min1/uvu2_min1/dualcut2_min1
(trained ≤2026-07-29): GeO −6.4138, TiO −8.946, TiGeO −7.559, HfO −11.090,
SiO −7.864, Si −5.424, NbO −9.2824, ZrTaO −9.5356.
Asking for targets below the trained floor is allowed and useful (it drives the
model to its lowest-energy manifold — how the records were found).

`filter_trainset.py` E_REF_TRAIN uses min(table, observed-data minimum), so new
record structures self-consistently lower the label floors when the AL6 corpus is
rebuilt. The SiO₂ quartz floor (−7.9045) and ZrTaO tie-line (−9.6795) are
deliberately NOT pushed into E_REF_TRAIN yet — do that at the AL6 full-corpus
rebuild so every source pickle gets consistent labels in one shot.
