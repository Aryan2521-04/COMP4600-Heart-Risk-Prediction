# Heart Risk Prediction — COMP 4600

Predicting 10-year coronary heart disease (CHD) risk using the **Framingham Heart Study** dataset. Four machine learning models are trained, tuned, and compared using a **70 / 15 / 15** train / validation / test split.

## Models

| Model               | Owner  | Notes                           |
| ------------------- | ------ | ------------------------------- |
| Logistic Regression | Vin    | Baseline linear model           |
| Random Forest       | Aryan  | Ensemble, handles non-linearity |
| SVM (RBF kernel)    | Elyas  | Kernel-based classification     |
| XGBoost             | Matias | Gradient boosted trees + SHAP   |

---

## Repository Structure

```
.
├── data/
│   ├── raw/            # Place Framingham dataset here (gitignored)
│   └── processed/      # Cleaned and split data (train.csv, val.csv, test.csv)
├── models/             # Saved scaler (scaler.pkl) and trained model files
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

Download the Framingham Heart Study dataset from Kaggle and place it at:

```
data/raw/framingham.csv
```

Dataset link: https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset

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

## Preprocessing Pipeline (Already Complete — Do Not Re-Run)

The preprocessing pipeline has been completed and the processed data is already saved in `data/processed/`. **Do not re-run the preprocessing notebook** unless there is a specific reason to — doing so will overwrite the shared splits and break consistency across models.

### What was done

| Step                      | Details                                                                                        |
| ------------------------- | ---------------------------------------------------------------------------------------------- |
| Missing value imputation  | Median imputation applied to all columns with missing values (`glucose` had the most at 9.15%) |
| Stratified 70/15/15 split | `random_state=42`, `stratify=y` to preserve the 85/15 class ratio in every split               |
| Feature scaling           | `StandardScaler` fitted **only on training data**, then applied to val and test                |
| Class imbalance (SMOTE)   | Applied to training set only — val and test remain the real unbalanced distribution            |
| Saved outputs             | `train.csv`, `val.csv`, `test.csv` in `data/processed/` and `scaler.pkl` in `models/`          |

### Key findings from EDA

- **Class imbalance**: ~85% negative (no CHD), ~15% positive (CHD risk) — use AUC-ROC and F1 as primary metrics, not accuracy
- **Top predictive features**: age (0.225), sysBP (0.216), prevalentHyp (0.177), diaBP (0.145), glucose (0.126)
- **Weak features**: heartRate, currentSmoker, education — near-zero correlation with target (kept in for tree-based models)

---

## How to Load the Processed Data (All Team Members)

Every team member should load the preprocessed data the same way. Do **not** load from `data/raw/` for model training.

```python
import pandas as pd
import joblib

# Load processed splits
train = pd.read_csv("../data/processed/train.csv")
val   = pd.read_csv("../data/processed/val.csv")
test  = pd.read_csv("../data/processed/test.csv")

# Separate features and target
X_train, y_train = train.drop(columns=["TenYearCHD"]), train["TenYearCHD"]
X_val,   y_val   = val.drop(columns=["TenYearCHD"]),   val["TenYearCHD"]
X_test,  y_test  = test.drop(columns=["TenYearCHD"]),  test["TenYearCHD"]

# Load scaler (only needed if you want to inverse-transform or scale new data)
scaler = joblib.load("../models/scaler.pkl")
```

> **Important**: `train.csv` already includes SMOTE-resampled data and is scaled. `val.csv` and `test.csv` are scaled but NOT resampled — they reflect the real class distribution for honest evaluation.

---

## Evaluation — Use the Same Metrics for Every Model

All four models must be evaluated using the same metrics on the same `test.csv` so results are directly comparable.

```python
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    recall_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"=== {name} ===")
    print(f"Accuracy:    {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC:     {roc_auc_score(y_test, y_prob):.4f}")
    print(f"F1 Score:    {f1_score(y_test, y_pred):.4f}")
    print(f"Sensitivity: {recall_score(y_test, y_pred):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix — {name}")
    plt.savefig(f"../results/{name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.show()
```

---

## Success Metrics

| Metric               | Target                                                       |
| -------------------- | ------------------------------------------------------------ |
| Accuracy             | > 80%                                                        |
| AUC-ROC              | > 0.75                                                       |
| F1 Score             | As high as possible                                          |
| Sensitivity (Recall) | Prioritize — missing an at-risk patient is the worst outcome |

**Qualitative goals:**

- Model explainability via SHAP (XGBoost)
- Fair performance across demographic subgroups where possible

---

## Week-by-Week Plan (April 1 – April 29)

| Week      | Dates     | Goals                                                                               |
| --------- | --------- | ----------------------------------------------------------------------------------  |
| **1**     | Apr 1–7   | ✅ Repo setup, EDA, data exploration                                                |
| **2**     | Apr 8–14  | ✅ Preprocessing complete — imputation, scaling, SMOTE, train/val/test split saved  |
| **3**     | Apr 15–21 | ✅ Individual model implementation and hyperparameter tuning (each member)          |
| **4**     | Apr 22–28 | Model comparison, SHAP analysis, final presentation slides and writeup               |
| **Final** | Apr 29    | Final presentation                                                                   |

---

## Team

| Member               | Model               | Role                                           |
| -------------------- | ------------------- | ---------------------------------------------- |
| Aryan Shah (Owner)   | Random Forest       | Ensemble model + feature importance            |
| Elyas Khoubach       | SVM                 | Kernel model + confusion matrix visualizations |
| Matias Elban Stringa | XGBoost             | Refined model + SHAP explainability analysis   |
| Vin Patel            | Logistic Regression | Baseline model + data preprocessing            |

---

## Dataset

**Framingham Heart Study** — a longitudinal cardiovascular study originally conducted by Boston University. Target variable: `TenYearCHD` (binary, 10-year risk of coronary heart disease).

- Source: https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset
- Citation: Kaggle / National Heart, Lung, and Blood Institute (NHLBI)
