import json
from pathlib import Path


def make_cell(cell_type: str, text: str):
    src = [line + "\n" for line in text.strip().splitlines()]
    base = {"cell_type": cell_type, "metadata": {}, "source": src}
    if cell_type == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


cells_specs = [
    (
        "markdown",
        """
# MIMIC-III Mortality Pipeline (full stack)
Crisp walk from EDA -> feature engineering -> preprocessing -> modeling -> submission.
        """,
    ),
    (
        "markdown",
        """
**Run notes**
- Toggle `EDA = True` for plots; keep `False` for speed.
- Data lives in `MIMIC III dataset HEF` with diagnoses in `extra_data`.
- Output CSV: `argote_mariajose_CML_2025.csv` with mortality probabilities.
        """,
    ),
    (
        "code",
        """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")
        """,
    ),
    (
        "code",
        """
RANDOM_STATE = 42
DATA_DIR = Path("MIMIC III dataset HEF")
TRAIN_PATH = DATA_DIR / "mimic_train_HEF.csv"
TEST_PATH = DATA_DIR / "mimic_test_HEF.csv"
DIAGNOSES_PATH = DATA_DIR / "extra_data" / "MIMIC_diagnoses.csv"

EDA = True  # set False to skip plots
pd.set_option("display.max_columns", None)
        """,
    ),
    (
        "code",
        """
# Load core data
train_raw = pd.read_csv(TRAIN_PATH)
test_raw = pd.read_csv(TEST_PATH)
diagnoses_raw = pd.read_csv(DIAGNOSES_PATH)

print(f"train shape: {train_raw.shape}")
print(f"test shape:  {test_raw.shape}")
print(f"diagnoses shape: {diagnoses_raw.shape}")
train_raw.head()
        """,
    ),
    (
        "markdown",
        """
### Quick audit and missingness
        """,
    ),
    (
        "code",
        """
train_cols = set(train_raw.columns)
test_cols = set(test_raw.columns)
missing_in_test = train_cols - test_cols
print("Columns present in train but missing in test:", missing_in_test)

missing_cols = train_raw.isna().mean().sort_values(ascending=False)
missing_cols.head(10)
        """,
    ),
    (
        "code",
        """
if EDA:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    train_raw["HOSPITAL_EXPIRE_FLAG"].value_counts(normalize=True).rename("freq").plot(kind="bar", ax=ax[0], color=["steelblue", "salmon"])
    ax[0].set_title("Target distribution")
    train_raw[[c for c in train_raw.columns if "_Min" in c or "_Max" in c or "_Mean" in c]].isna().sum(axis=1).plot(kind="hist", bins=30, ax=ax[1])
    ax[1].set_title("Missing vital summaries per row")
    plt.tight_layout()
        """,
    ),
    (
        "markdown",
        """
### Drop rows with extreme vital missingness
Rows with >=21 of 24 vital summaries missing add noise.
        """,
    ),
    (
        "code",
        """
VITAL_COLS = [
    "HeartRate_Min", "HeartRate_Max", "HeartRate_Mean",
    "SysBP_Min", "SysBP_Max", "SysBP_Mean",
    "DiasBP_Min", "DiasBP_Max", "DiasBP_Mean",
    "MeanBP_Min", "MeanBP_Max", "MeanBP_Mean",
    "RespRate_Min", "RespRate_Max", "RespRate_Mean",
    "TempC_Min", "TempC_Max", "TempC_Mean",
    "SpO2_Min", "SpO2_Max", "SpO2_Mean",
    "Glucose_Min", "Glucose_Max", "Glucose_Mean",
]

def drop_sparse_vitals(df: pd.DataFrame, threshold: int = 21) -> pd.DataFrame:
    df = df.copy()
    missing_counts = df[VITAL_COLS].isna().sum(axis=1)
    keep_mask = missing_counts < threshold
    print(f"Dropping {(~keep_mask).sum()} rows with >= {threshold} missing vital features")
    return df.loc[keep_mask].reset_index(drop=True)

train_clean = drop_sparse_vitals(train_raw)
test_clean = test_raw.copy()
print("train_clean shape:", train_clean.shape)
        """,
    ),
    (
        "markdown",
        """
### Age feature
        """,
    ),
    (
        "code",
        """
from pandas import to_datetime

def add_age_at_admission(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["DOB", "ADMITTIME"]:
        df[col] = to_datetime(df[col], errors="coerce")
    df["age_at_admission"] = (df["ADMITTIME"] - df["DOB"]).dt.total_seconds() / (60 * 60 * 24 * 365.25)
    df["age_at_admission"] = df["age_at_admission"].clip(lower=0, upper=110)
    return df

train_enriched = add_age_at_admission(train_clean)
test_enriched = add_age_at_admission(test_clean)
train_enriched[["age_at_admission", "HOSPITAL_EXPIRE_FLAG"]].describe()
        """,
    ),
    (
        "markdown",
        """
### Diagnosis severity features (Bayesian smoothing)
        """,
    ),
    (
        "code",
        """
def fit_diagnosis_features(train_df: pd.DataFrame, diagnoses_df: pd.DataFrame, alpha: int = 10):
    df = train_df.copy()
    diag = diagnoses_df.copy()
    diag["HADM_ID"] = diag["HADM_ID"].astype(int)
    diag["SEQ_NUM"] = diag["SEQ_NUM"].astype(float)

    merged = diag.merge(df[["hadm_id", "HOSPITAL_EXPIRE_FLAG"]], left_on="HADM_ID", right_on="hadm_id", how="left")
    merged = merged.dropna(subset=["HOSPITAL_EXPIRE_FLAG"])

    overall_mortality = merged["HOSPITAL_EXPIRE_FLAG"].mean()
    by_code = merged.groupby("ICD9_CODE")["HOSPITAL_EXPIRE_FLAG"].agg(["mean", "count"])
    by_code["severity"] = (by_code["count"] * by_code["mean"] + alpha * overall_mortality) / (by_code["count"] + alpha)
    severity_map = by_code["severity"].to_dict()

    merged["severity_score"] = merged["ICD9_CODE"].map(severity_map).fillna(overall_mortality)

    agg = merged.groupby("HADM_ID").agg(
        diagnosis_count=("ICD9_CODE", "count"),
        avg_diagnosis_severity=("severity_score", "mean"),
        max_diagnosis_severity=("severity_score", "max"),
    )

    primary = merged.loc[merged["SEQ_NUM"] == 1, ["HADM_ID", "severity_score"]].rename(
        columns={"severity_score": "primary_diagnosis_severity"}
    )
    agg = agg.merge(primary, on="HADM_ID", how="left")
    agg["primary_diagnosis_severity"] = agg["primary_diagnosis_severity"].fillna(overall_mortality)

    df = df.merge(agg, left_on="hadm_id", right_on="HADM_ID", how="left")
    df["diagnosis_count"] = df["diagnosis_count"].fillna(0)
    for col in ["avg_diagnosis_severity", "max_diagnosis_severity", "primary_diagnosis_severity"]:
        df[col] = df[col].fillna(overall_mortality)

    state = {"severity_map": severity_map, "overall_mortality": overall_mortality, "alpha": alpha}
    return df.drop(columns=["HADM_ID"], errors="ignore"), state


def apply_diagnosis_features(df: pd.DataFrame, diagnoses_df: pd.DataFrame, state: dict):
    df = df.copy()
    diag = diagnoses_df.copy()
    diag["HADM_ID"] = diag["HADM_ID"].astype(int)
    diag["SEQ_NUM"] = diag["SEQ_NUM"].astype(float)

    diag["severity_score"] = diag["ICD9_CODE"].map(state["severity_map"]).fillna(state["overall_mortality"])

    agg = diag.groupby("HADM_ID").agg(
        diagnosis_count=("ICD9_CODE", "count"),
        avg_diagnosis_severity=("severity_score", "mean"),
        max_diagnosis_severity=("severity_score", "max"),
    )

    primary = diag.loc[diag["SEQ_NUM"] == 1, ["HADM_ID", "severity_score"]].rename(
        columns={"severity_score": "primary_diagnosis_severity"}
    )
    agg = agg.merge(primary, on="HADM_ID", how="left")
    agg["primary_diagnosis_severity"] = agg["primary_diagnosis_severity"].fillna(state["overall_mortality"])

    df = df.merge(agg, left_on="hadm_id", right_on="HADM_ID", how="left")
    df["diagnosis_count"] = df["diagnosis_count"].fillna(0)
    for col in ["avg_diagnosis_severity", "max_diagnosis_severity", "primary_diagnosis_severity"]:
        df[col] = df[col].fillna(state["overall_mortality"])

    return df.drop(columns=["HADM_ID"], errors="ignore")
        """,
    ),
    (
        "code",
        """
train_diag, diag_state = fit_diagnosis_features(train_enriched, diagnoses_raw)
print("Overall mortality used for smoothing:", diag_state["overall_mortality"])

test_diag = apply_diagnosis_features(test_enriched, diagnoses_raw, diag_state)
train_diag[["diagnosis_count", "avg_diagnosis_severity", "max_diagnosis_severity", "primary_diagnosis_severity"]].describe()
        """,
    ),
    (
        "markdown",
        """
### Preprocessing (impute -> encode -> scale)
        """,
    ),
    (
        "code",
        """
target_col = "HOSPITAL_EXPIRE_FLAG"
categorical_cols = [
    "GENDER", "ADMISSION_TYPE", "INSURANCE", "RELIGION",
    "MARITAL_STATUS", "ETHNICITY", "FIRST_CAREUNIT",
]
drop_cols = [
    "subject_id", "hadm_id", "icustay_id",
    "DOB", "ADMITTIME", "DISCHTIME", "DOD", "DEATHTIME", "LOS",
    "DIAGNOSIS", "ICD9_diagnosis",
]
        """,
    ),
    (
        "code",
        """
def fit_preprocessing_pipeline(df: pd.DataFrame, target_col: str):
    df = df.copy()
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")

    cat_cols = [c for c in categorical_cols if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ],
        remainder="drop",
    )

    X_processed = preprocessor.fit_transform(X)

    feature_names = []
    if len(cat_cols) > 0:
        enc = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        feature_names.extend(enc.get_feature_names_out(cat_cols))
    feature_names.extend(num_cols)

    state = {"preprocessor": preprocessor, "feature_names": feature_names, "cat_cols": cat_cols, "num_cols": num_cols}
    return X_processed, y, state


def apply_preprocessing_pipeline(df: pd.DataFrame, state: dict):
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    preprocessor = state["preprocessor"]
    return preprocessor.transform(X)
        """,
    ),
    (
        "code",
        """
X_train, y_train, preprocess_state = fit_preprocessing_pipeline(train_diag, target_col)
print("Processed train shape:", X_train.shape)
print("Positive rate:", y_train.mean())
        """,
    ),
    (
        "markdown",
        """
### Baseline models and cross-validation
        """,
    ),
    (
        "code",
        """
def evaluate_models_cv(models: dict, X, y, cv_splits: int = 5):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, model in models.items():
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        pr_scores = cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=-1)
        results[name] = {
            "roc_auc_mean": auc_scores.mean(),
            "roc_auc_std": auc_scores.std(),
            "pr_auc_mean": pr_scores.mean(),
            "pr_auc_std": pr_scores.std(),
        }
    return pd.DataFrame(results).T.sort_values(by="roc_auc_mean", ascending=False)
        """,
    ),
    (
        "code",
        """
imbalance_weight = (y_train == 0).sum() / (y_train == 1).sum()
models = {
    "log_reg": LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=-1),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

if HAS_XGB:
    models["xgboost"] = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=float(imbalance_weight),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
else:
    print("xgboost not installed in this env. Install if you want that model: pip install xgboost")
        """,
    ),
    (
        "code",
        """
cv_results = evaluate_models_cv(models, X_train, y_train)
cv_results
        """,
    ),
    (
        "markdown",
        """
### Optional: quick XGBoost grid search (flip flag to run)
        """,
    ),
    (
        "code",
        """
RUN_GRIDSEARCH = False
if RUN_GRIDSEARCH and HAS_XGB:
    param_grid = {
        "n_estimators": [150, 250, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    xgb = models["xgboost"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    print("Best params:", grid.best_params_)
    print("Best ROC-AUC:", grid.best_score_)
    best_estimator = grid.best_estimator_
else:
    best_estimator = None
        """,
    ),
    (
        "markdown",
        """
### Final model fit on full training data
        """,
    ),
    (
        "code",
        """
if best_estimator is None:
    best_model_name = cv_results.index[0]
    best_model = clone(models[best_model_name])
else:
    best_model_name = "xgboost_tuned"
    best_model = best_estimator

best_model.fit(X_train, y_train)
print(f"Fitted {best_model_name} on full training set")
        """,
    ),
    (
        "markdown",
        """
### Feature importance peek
        """,
    ),
    (
        "code",
        """
feature_names = preprocess_state.get("feature_names", [])

if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=feature_names)
    display(importances.sort_values(ascending=False).head(20))
elif hasattr(best_model, "coef_"):
    coefs = pd.Series(best_model.coef_[0], index=feature_names)
    display(coefs.sort_values(key=abs, ascending=False).head(20))
else:
    print("Model does not expose feature importances directly.")
        """,
    ),
    (
        "markdown",
        """
### Test-set preprocessing and submission file
        """,
    ),
    (
        "code",
        """
X_test = apply_preprocessing_pipeline(test_diag, preprocess_state)
test_predictions = best_model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "icustay_id": test_diag["icustay_id"],
    "prediction": test_predictions,
})
submission.to_csv("argote_mariajose_CML_2025.csv", index=False)

print(submission.head())
print("Saved submission -> argote_mariajose_CML_2025.csv")
        """,
    ),
    (
        "markdown",
        """
### Next steps
- Run top-to-bottom and log CV metrics in markdown cells for the deck
- Push submission and capture leaderboard score
        """,
    ),
]

cells = [make_cell(kind, text) for kind, text in cells_specs]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = Path("Argote_Mariajose_CML_pipeline.ipynb")
output_path.write_text(json.dumps(nb, indent=2))
print(f"Notebook written to {output_path.resolve()}")
