# Output Figures and Tables

All files in this directory are **generated automatically** by running `thesis_simulation.py`.

## Manuscript Figures (Figs 1–8 → main text; Fig 9 → supplementary)

| File | Description | Manuscript section |
|---|---|---|
| `fig1_schematic.png` | Spray dryer geometry (Büchi B-290 / Niro Mobile Minor) | Materials & Methods |
| `fig2_thermal.png` | Particle thermal history — two-stage Lagrangian model | Materials & Methods |
| `fig3_psd.png` | Particle size distribution (log-normal, 300 particles) | Results |
| `fig4_AA_T.png` | AA retention vs T_in with Monte Carlo error bars + split validation | Results |
| `fig5_surface.png` | RSM 3D response surface (AA vs T_in and Q) | Results |
| `fig6_validation.png` | Model validation — calibration set vs test set (n=9 total) | Results |
| `fig7_anova.png` | ANOVA significance + standardized coefficients | Results |
| `fig8_contour.png` | Contour plots with 95% Monte Carlo confidence intervals | Results |

## Dashboard / Supplementary Figures

| File | Description |
|---|---|
| `fig9_dashboard.png` | Dashboard: Tg analysis panel + Pareto front (AA vs d) + Morris chart |
| `figS1_arrhenius_MC.png` | Supplementary S1: Arrhenius k(T) with Monte Carlo envelope |
| `figS2_monte_carlo.png` | Supplementary S2: Monte Carlo AA retention distribution at optimal |
| `figS3_glass_transition.png` | Supplementary S3: Tg vs moisture content (Gordon-Taylor + Couchman-Karasz) |
| `figS4_ranz_marshall.png` | Supplementary S4: Ranz-Marshall Nu/Sh correlation validation |
| `figS5_morris.png` | Supplementary S5: Morris sensitivity µ* vs σ scatter plot |

## Data Tables

| File | Description |
|---|---|
| `table_BBD_results.csv` | 15 Box-Behnken runs: T_in, Q, Cs → AA_ret, diameter, T_out, Tg, T-Tg |
| `table_validation.csv` | 9 validation points: source, T_in, AA_exp, AA_pred, error |
| `table_ANOVA.csv` | Full ANOVA table: source, SS, df, MS, F, p-value |
| `table_MC_uncertainty.csv` | Monte Carlo summary: mean, std, CI_lower, CI_upper per BBD run |
| `table_sensitivity.csv` | Morris sensitivity indices: µ, µ*, σ per factor |

## Regenerating All Outputs

```bash
python thesis_simulation.py
```

Expected runtime: ~2–5 minutes on a standard laptop.
