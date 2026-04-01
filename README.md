# Heart Risk Prediction — COMP 4600

Predicting 10-year coronary heart disease (CHD) risk using the **Framingham Heart Study** dataset. Four machine learning models are trained, tuned, and compared using a **70 / 15 / 15** train / validation / test split.

## Models

| Model               | Notes                           |
| ------------------- | ------------------------------- |
| Logistic Regression | Baseline linear model           |
| Random Forest       | Ensemble, handles non-linearity |
| SVM (RBF kernel)    | Kernel-based classification     |
| XGBoost             | Gradient boosted trees          |

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

## Cloning the Repo (Collaborators)

> You must be added as a collaborator by Aryan before you can clone or push.

```bash
git clone git@github.com:Aryan2521-04/COMP4600-Heart-Risk-Prediction.git
cd COMP4600-Heart-Risk-Prediction
```

Then follow **Option A** or **Option B** above to set up your environment.

---

## Git Workflow

Follow this workflow to avoid conflicts on `main`:

1. **Pull the latest changes before starting any work**
   ```bash
   git pull origin main
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b your-name/feature-name
   # e.g. git checkout -b aryan/eda-notebook
   ```

3. **Stage and commit your changes**
   ```bash
   git add <files>
   git commit -m "short description - YourName"
   ```

4. **Push your branch**
   ```bash
   git push origin your-name/feature-name
   ```

5. **Open a Pull Request on GitHub** to merge your branch into `main`

> Avoid pushing directly to `main` unless it's a trivial fix (typo, config update, etc.).

---

## Week-by-Week Plan

| Week  | Goals                                                                                        |
| ----- | -------------------------------------------------------------------------------------------- |
| **1** | Repo setup, dataset acquisition, exploratory data analysis (EDA)                             |
| **2** | Data cleaning, handling missing values, 70/15/15 split pipeline                              |
| **3** | Feature engineering, baseline Logistic Regression model                                      |
| **4** | Random Forest — implementation, hyperparameter tuning                                        |
| **5** | SVM with RBF kernel — implementation, hyperparameter tuning                                  |
| **6** | XGBoost — implementation, hyperparameter tuning                                              |
| **7** | Model comparison: AUC-ROC, F1, precision/recall, confusion matrices, SHAP feature importance |
| **8** | Final writeup, presentation slides, repo cleanup and documentation                           |

---

## Team

- Member 1 Aryan (Owner)
- Member 2 Matias
- Member 3 Vin
- Member 4 Elyas

---

## Dataset

**Framingham Heart Study** — a longitudinal cardiovascular study. Target variable: `TenYearCHD` (binary, 10-year risk of coronary heart disease).
