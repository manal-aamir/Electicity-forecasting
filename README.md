# HW + XGBoost + XAI (SHAP/LIME) for Multi-Scale Household Energy Forecasting

This project implements the paper's **HW+XGB+XAI** pipeline on the **UCI Individual Household Electric Power Consumption** dataset:

- **Multi-scale** forecasting: hourly, daily, weekly, monthly, quarterly
- **Targets**: global active power (default) and per-appliance sub-metering (optional)
- **Model**: Holt-Winters feature augmentation + XGBoost regressor
- **XAI**: SHAP, LIME, Permutation Feature Importance (PFI), Partial Dependence Plots (PDP)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (global active power)

```bash
python run_hw_xgb_xai.py --scales hourly daily weekly monthly quarterly --target global_active_power
```

## Run (appliance-level targets)

```bash
python run_hw_xgb_xai.py --scales hourly daily weekly monthly quarterly --target sub_metering_1 sub_metering_2 sub_metering_3
```

## Outputs

Artifacts are written to `outputs/`:

- `metrics.csv`: MAE/MSE/RMSE/R2 per (scale,target)
- `models/`: serialized models
- `plots/`: SHAP summary, PDP, etc.
- `explanations/`: per-instance LIME explanations (HTML)

## Notes (implementation choices)

- Chronological split (default 80/20), no shuffling.
- Time-aware CV uses `TimeSeriesSplit`.
- Holt-Winters is used for **feature augmentation**. To avoid leakage, Holt-Winters components are **lagged by 1 step** when predicting the next step.

## Makefile shortcuts

```bash
make help
make install
make run-all-scales
make paper          # build IEEE_paper_draft.pdf (needs LaTeX)
```

## Publish to GitHub

1. Create an **empty** repository on GitHub (no README/license) — e.g. `household-energy-forecasting-xai`.
2. From this project folder:

```bash
git init
git add .
git commit -m "Initial commit: HW+XGB+XAI pipeline, error decomposition, ablation"
git branch -M main
git remote add origin https://github.com/manal-aamir/Electicity-forecasting.git
git push -u origin main
```

If you use [GitHub CLI](https://cli.github.com/):

```bash
gh auth login
gh repo create YOUR_REPO --public --source=. --remote=origin --push
```

Large files (`data/raw/*.txt`, `outputs/`, reference PDF) are listed in `.gitignore`; collaborators run the pipeline to regenerate outputs.
# Electicity-forecasting
