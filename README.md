# Heart Risk Prediction — COMP 4600

Predicting 10-year coronary heart disease (CHD) risk using the **Framingham Heart Study** dataset. Four machine learning models are trained, tuned, and compared using a **70 / 15 / 15** train / validation / test split.

## Models
| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble, handles non-linearity |
| SVM (RBF kernel) | Kernel-based classification |
| XGBoost | Gradient boosted trees |

---

## Repository Structure
```
.
├── data/
│   ├── raw/            # Place Framingham dataset here (gitignored)
│   └── processed/      # Cleaned and split data
├── notebooks/          # Exploratory data analysis
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── features/       # Feature engineering
│   ├── models/         # One module per model
│   └── utils/          # Shared helpers (metrics, plotting)
├── results/            # Saved metrics, plots, outputs
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Getting Started

### Option A — Conda (recommended)
```bash
conda env create -f environment.yml
conda activate heart-risk
```

### Option B — pip
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Register the kernel (for Jupyter)
```bash
python -m ipykernel install --user --name heart-risk --display-name "Heart Risk"
jupyter notebook
```

### Add the dataset
Download the Framingham Heart Study dataset and place it at:
```
data/raw/framingham.csv
```

---

## Week-by-Week Plan

| Week | Goals |
|------|-------|
| **1** | Repo setup, dataset acquisition, exploratory data analysis (EDA) |
| **2** | Data cleaning, handling missing values, 70/15/15 split pipeline |
| **3** | Feature engineering, baseline Logistic Regression model |
| **4** | Random Forest — implementation, hyperparameter tuning |
| **5** | SVM with RBF kernel — implementation, hyperparameter tuning |
| **6** | XGBoost — implementation, hyperparameter tuning |
| **7** | Model comparison: AUC-ROC, F1, precision/recall, confusion matrices, SHAP feature importance |
| **8** | Final writeup, presentation slides, repo cleanup and documentation |

---

## Team
- Member 1 (repo owner)
- Member 2
- Member 3
- Member 4

---

## Dataset
**Framingham Heart Study** — a longitudinal cardiovascular study. Target variable: `TenYearCHD` (binary, 10-year risk of coronary heart disease).
