# Lagrangian CFD Simulation of Ascorbic Acid Retention During Spray Drying of *Myrciaria dubia* (Camu Camu) Extract

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey.svg)](#citation)
[![Journal](https://img.shields.io/badge/Target-Journal%20of%20Food%20Engineering%20(Q1)-orange.svg)](#)

> **Tesis por Artículo** — Ingeniería de Industrias Alimentarias  
> Universidad Nacional de San Agustín de Arequipa (UNSA), 2026

---

## Overview

This repository contains the full simulation code, results, and supporting materials for a **Lagrangian particle-tracking CFD model** that predicts ascorbic acid (vitamin C) retention during spray drying of *Myrciaria dubia* (camu camu) extract.

The model integrates:
- **Arrhenius degradation kinetics** (Ea = 56.7 kJ/mol, Viera et al. 2000)
- **Two-stage Lagrangian particle thermal model** calibrated to Fujita et al. (2017) data
- **Box-Behnken Response Surface Methodology** (RSM) with 15 runs
- **Monte Carlo uncertainty quantification** (n = 5 000 samples)
- **Morris global sensitivity analysis**
- **Glass transition temperature (Tg) analysis** for stickiness risk

**Key finding:** Optimal conditions (T_in = 120 °C, Q = 15 mL/min, Cs = 15% w/w) yield **88.8% AA retention** with zero stickiness risk (T_out − Tg = −33 °C).

---

## Repository Structure

```
.
├── thesis_simulation.py          # Main simulation code (~1 000 lines)
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
├── CITATION.cff                  # Citation metadata
├── README.md                     # This file
│
├── outputs/                      # Generated figures and tables
│   ├── fig1_schematic.png        # Spray dryer geometry
│   ├── fig2_thermal.png          # Particle thermal history
│   ├── fig3_psd.png              # Particle size distribution
│   ├── fig4_AA_T.png             # AA retention vs temperature (+ MC error bars)
│   ├── fig5_surface.png          # RSM response surface
│   ├── fig6_validation.png       # Model validation (calibration vs test)
│   ├── fig7_anova.png            # ANOVA significance + coefficients
│   ├── fig8_contour.png          # Contour plots with MC confidence intervals
│   ├── fig9_dashboard.png        # Dashboard: Tg panel + Pareto front + Morris
│   ├── figS1_arrhenius_MC.png    # Supplementary: Arrhenius Monte Carlo
│   ├── figS2_monte_carlo.png     # Supplementary: MC uncertainty distribution
│   ├── figS3_glass_transition.png# Supplementary: Tg analysis
│   ├── figS4_ranz_marshall.png   # Supplementary: Ranz-Marshall validation
│   ├── figS5_morris.png          # Supplementary: Morris sensitivity
│   ├── table_BBD_results.csv     # Box-Behnken design + responses
│   ├── table_validation.csv      # Validation data (9 independent points)
│   ├── table_ANOVA.csv           # Full ANOVA table
│   ├── table_MC_uncertainty.csv  # Monte Carlo summary statistics
│   └── table_sensitivity.csv     # Morris sensitivity indices
│
└── manuscript_Q1_blueprint.md    # Section-by-section manuscript guide
```

---

## Scientific Background

### Why Computational Modeling?

No spray dryer is available at UNSA. A validated Lagrangian CFD approach is well-established in literature (Mezhericher et al. 2010; Langrish & Fletcher 2001) and scientifically defensible when:
1. The model is calibrated against published experimental data
2. Validation is performed against independent datasets (n ≥ 6, multiple sources)
3. Uncertainty is quantified explicitly

### Ascorbic Acid Degradation Model

```
dC/dt = −k(T) · C
k(T)  = k₀ · exp(−Ea / RT)

Ea = 56 700 J/mol   [Viera et al. 2000]
k₀ = 8.0 × 10⁷ s⁻¹ [calibrated vs. Fujita et al. 2017]
```

### Effective Particle Temperature (Lagrangian Model)

Calibrated against Fujita et al. (2017) camu camu spray drying data (n = 5, R² = 0.997):

```
T_eff (°C) = 0.278·T_in + 13.94 − 0.14·(Q − 10) − 0.10·(Cs − 20)
t_eff  (s) = 2.5 + 0.08·(Q − 10) + 0.02·(Cs − 20)
```

### Spray Dryer Geometry (Büchi B-290 / Niro Mobile Minor)

| Parameter | Value |
|---|---|
| Chamber diameter | 0.50 m |
| Cylindrical height | 0.85 m |
| Conical height | 0.50 m |
| Cone angle | 60° |
| Air flow rate | 35 m³/h |
| Nozzle diameter | 1.5 mm |
| Particles per run | 300 |

---

## Box-Behnken Design

| Factor | Low (−1) | Center (0) | High (+1) |
|---|---|---|---|
| T_in (°C) | 120 | 150 | 180 |
| Q (mL/min) | 5 | 10 | 15 |
| Cs (% w/w) | 15 | 20 | 25 |

**15 runs** with 3 center points. Responses: AA retention (%) and mean particle diameter (µm).

---

## Key Results

### RSM Model Quality

| Metric | Value |
|---|---|
| R² (AA retention) | **0.9976** |
| R²_adj | **0.9933** |
| R²_pred | **0.9697** |
| RMSE | 0.31% |
| ANOVA F (model) | 230.99 |
| ANOVA p (model) | < 0.0001 |
| Lack-of-Fit p | 0.3426 (NS — model adequate) |

### Optimal Conditions

| Parameter | Value |
|---|---|
| T_in | **120 °C** |
| Q | **15 mL/min** |
| Cs | **15% w/w** |
| Predicted AA retention | **88.8%** |
| Mean particle diameter | **43.3 µm** |
| Pareto-optimal solutions | 15 |

### Validation (Independent Data, n = 9)

| Metric | All (n=9) | Test set (n=6) |
|---|---|---|
| MAE | **3.00%** | **3.35%** |
| RMSE | — | **3.83%** |

> Note: Pearson r is not the primary metric due to the narrow experimental range (75–84%); MAE < inter-study variability confirms model adequacy.

### Uncertainty Quantification (Monte Carlo, n = 5 000)

| Condition | AA Retention |
|---|---|
| Nominal (optimal) | 88.8% |
| 95% CI | [75.9, 94.3]% |
| 1σ uncertainty | ±4.75% |

The CI reflects parametric sensitivity (Ea ± 1 kJ/mol), not classical statistical uncertainty.

### Sensitivity Analysis (Morris Method)

| Factor | Global (Morris µ*) | Local (ANOVA) |
|---|---|---|
| T_in | µ* = 0.311 | t = −45.13, p < 0.0001 |
| Q | **µ* = 0.394 (dominant)** | t = −3.21 |
| Cs | µ* = 0.089 | t = −1.44 |

T_in dominates locally (direct thermal effect); Q dominates globally (affects residence time more at high temperatures).

### Glass Transition Analysis

| Condition | Tg (°C) | T_out (°C) | Margin |
|---|---|---|---|
| Optimal (T=120°C, Cs=15%) | ~115 | ~82 | **−33 °C (safe)** |
| Risky runs (T_in=180°C) | ~105 | ~140 | +35 °C (sticky risk) |

4 of 15 runs show stickiness risk; all at T_in = 180 °C. The optimal conditions eliminate this risk.

---

## Installation

```bash
git clone https://github.com/<your-username>/camu-camu-spray-drying.git
cd camu-camu-spray-drying
pip install -r requirements.txt
```

### Requirements

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
pandas>=2.0
```

---

## Usage

```bash
python thesis_simulation.py
```

All figures (Figs 1–9 + S1–S5) and tables (CSV) are saved to the `outputs/` directory.

**Expected runtime:** ~2–5 minutes on a standard laptop (Monte Carlo n = 5 000, 15 BBD runs × 300 particles each).

**Verification after any code change:**
- AA retention range should remain ~69–89%
- RSM R² should remain > 0.99

---

## Kinetic Parameters (DO NOT CHANGE without bibliographic justification)

```python
Ea  = 56_700  # J/mol — Viera et al. (2000) J Food Eng 46:269
k0  = 8.0e7   # s^-1  — calibrated vs. Fujita et al. (2017) J Food Sci 82:1083
R   = 8.314   # J/(mol·K)
```

---

## References

1. Viera, M.C., Teixeira, A.A., & Silva, C.L.M. (2000). Mathematical modeling of the thermal degradation kinetics of vitamin C in whole tomato. *Journal of Food Engineering*, 46(4), 269–279.
2. Fujita, A., et al. (2017). Effects of spray-drying conditions on antioxidant capacity and volatile compounds of camu-camu. *Journal of Food Science*, 82(5), 1083–1091. **[Calibration reference]**
3. Mezhericher, M., Levy, A., & Borde, I. (2010). Theoretical models of single droplet drying kinetics. *Drying Technology*, 28(2), 278–293.
4. Langrish, T.A.G., & Fletcher, D.F. (2001). Spray drying of food ingredients and applications of CFD in spray drying. *Chemical Engineering and Processing*, 40(4), 345–354.
5. Masters, K. (1991). *Spray Drying Handbook* (5th ed.). Longman Scientific & Technical.
6. Southwell, D.B., & Langrish, T.A.G. (2000). Observations of flow patterns in a spray dryer. *Drying Technology*, 18(3), 661–685.
7. Roos, Y., & Karel, M. (1991). Amorphous state and delayed ice formation in sucrose solutions. *International Journal of Food Science & Technology*, 26(6), 553–566.
8. Angell, C.A. (2002). Liquid fragility and the glass transition in water and aqueous solutions. *Chemical Reviews*, 102(8), 2627–2650.
9. Morris, M.D. (1991). Factorial sampling plans for preliminary computational experiments. *Technometrics*, 33(2), 161–174.
10. Ranz, W.E., & Marshall, W.R. (1952). Evaporation from drops. *Chemical Engineering Progress*, 48(3), 141–146.

---

## Citation

If you use this code or results in your research, please cite:

```bibtex
@article{pcornejoaaguirre2026camucamu,
  title   = {Uncertainty-Quantified Lagrangian Simulation of Ascorbic Acid Retention
             During Spray Drying of \textit{Myrciaria dubia} Extract:
             Response Surface Optimization and Glass Transition Analysis},
  author  = {Priscila Cornejo, Alvaro Aguirre},
  journal = {Journal of Food Engineering},
  year    = {2026},
  note    = {Submitted}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The scientific parameters (Ea, k0) are derived from published literature and must not be modified without bibliographic justification.

---

## Author

**Priscila Cornejo, Alvaro Aguirre**  
Escuela Profesional de Ingeniería de Industrias Alimentarias  
Universidad Nacional de San Agustín de Arequipa (UNSA)  

---

## Acknowledgements

This work was developed as a thesis article (*tesis por artículo*) at UNSA. Computational resources were provided by the authors. The study was conducted under the supervision of [Advisor name], UNSA.

*Camu camu (Myrciaria dubia) is native to the Amazon basin and is one of the richest known sources of ascorbic acid (up to 6 000 mg/100 g fresh pulp).*
