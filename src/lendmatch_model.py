import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACT_PATH = MODELS_DIR / "lendmatch_artifacts.joblib"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"

NUMERIC_FEATURES = [
    "loan_amount",
    "annual_inc",
    "fico_score",
    "dti",
    "term",
    "emp_length_years",
    "revol_bal",
    "revol_util",
    "total_acc",
    "open_acc",
    "delinq_2yrs",
    "inq_last_6mths",
    "pub_rec",
    "credit_history_years",
]

CATEGORICAL_FEATURES = [
    "state",
    "purpose",
    "home_ownership",
    "verification_status",
    "application_type",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
APPROVAL_NUMERIC_FEATURES = ["loan_amount", "fico_score", "dti", "emp_length_years"]
APPROVAL_CATEGORICAL_FEATURES = ["state"]
APPROVAL_FEATURES = APPROVAL_NUMERIC_FEATURES + APPROVAL_CATEGORICAL_FEATURES

GOOD_STATUSES = {"fully paid"}
BAD_STATUS_PATTERNS = ("charged off", "default")


def parse_emp_length(value):
    if pd.isna(value):
        return 0.0
    text = str(value).strip().lower()
    if text in {"", "n/a", "nan", "unknown"}:
        return 0.0
    if "<" in text:
        return 0.5
    if "10+" in text:
        return 10.0
    match = re.search(r"\d+", text)
    return float(match.group(0)) if match else 0.0


def parse_term(value):
    if pd.isna(value):
        return 36.0
    match = re.search(r"\d+", str(value))
    return float(match.group(0)) if match else float(value or 36)


def parse_percent(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.replace("%", "").strip()
    return pd.to_numeric(value, errors="coerce")


def months_to_years_since(value, reference="2019-01-01"):
    dates = pd.to_datetime(value, format="%b-%Y", errors="coerce")
    ref = pd.Timestamp(reference)
    return ((ref - dates).dt.days / 365.25).clip(lower=0)


def clamp(value, low, high, fallback):
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return fallback
        return max(low, min(high, number))
    except (TypeError, ValueError):
        return fallback


def find_lendingclub_files(data_dir=DATA_DIR):
    candidates = [Path(data_dir), Path(data_dir) / "raw"]
    accepted = []
    rejected = []
    for folder in candidates:
        if not folder.exists():
            continue
        accepted.extend(folder.glob("*accepted*.csv"))
        rejected.extend(folder.glob("*rejected*.csv"))

    if not accepted or not rejected:
        raise FileNotFoundError(
            "Could not find LendingClub accepted/rejected CSVs in data/ or data/raw/."
        )

    def best(paths):
        return sorted(paths, key=lambda p: ("2007" in p.name and "2018" in p.name, p.stat().st_size), reverse=True)[0]

    return best(accepted), best(rejected)


def read_sample(path, usecols, max_rows, chunksize=150_000, seed=42):
    parts = []
    remaining = int(max_rows)
    rng = np.random.default_rng(seed)
    for chunk in pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=chunksize, low_memory=False):
        if remaining <= 0:
            break
        take = min(remaining, len(chunk))
        if take < len(chunk):
            chunk = chunk.sample(n=take, random_state=int(rng.integers(1_000_000_000)))
        parts.append(chunk)
        remaining -= len(chunk)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)


def normalize_accepted(df):
    out = pd.DataFrame()
    out["loan_amount"] = pd.to_numeric(df.get("loan_amnt"), errors="coerce")
    out["annual_inc"] = pd.to_numeric(df.get("annual_inc"), errors="coerce")
    fico_low = pd.to_numeric(df.get("fico_range_low"), errors="coerce")
    fico_high = pd.to_numeric(df.get("fico_range_high"), errors="coerce")
    out["fico_score"] = (fico_low + fico_high) / 2
    out["dti"] = pd.to_numeric(df.get("dti"), errors="coerce")
    out["term"] = df.get("term", 36).apply(parse_term) if "term" in df else 36
    out["emp_length_years"] = df.get("emp_length", "").apply(parse_emp_length) if "emp_length" in df else 0
    out["revol_bal"] = pd.to_numeric(df.get("revol_bal"), errors="coerce")
    out["revol_util"] = df.get("revol_util", np.nan).apply(parse_percent) if "revol_util" in df else np.nan
    out["total_acc"] = pd.to_numeric(df.get("total_acc"), errors="coerce")
    out["open_acc"] = pd.to_numeric(df.get("open_acc"), errors="coerce")
    out["delinq_2yrs"] = pd.to_numeric(df.get("delinq_2yrs"), errors="coerce")
    out["inq_last_6mths"] = pd.to_numeric(df.get("inq_last_6mths"), errors="coerce")
    out["pub_rec"] = pd.to_numeric(df.get("pub_rec"), errors="coerce")
    out["credit_history_years"] = months_to_years_since(df.get("earliest_cr_line")) if "earliest_cr_line" in df else np.nan
    out["state"] = df.get("addr_state", "Unknown").fillna("Unknown").astype(str)
    out["purpose"] = df.get("purpose", "other").fillna("other").astype(str)
    out["home_ownership"] = df.get("home_ownership", "Unknown").fillna("Unknown").astype(str)
    out["verification_status"] = df.get("verification_status", "Unknown").fillna("Unknown").astype(str)
    out["application_type"] = df.get("application_type", "Individual").fillna("Individual").astype(str)
    out["interest_rate"] = df.get("int_rate", np.nan).apply(parse_percent) if "int_rate" in df else np.nan

    status = df.get("loan_status", pd.Series(index=df.index, dtype=object)).fillna("").astype(str).str.lower()
    out["default_target"] = np.nan
    out.loc[status.isin(GOOD_STATUSES), "default_target"] = 0
    out.loc[status.apply(lambda s: any(pattern in s for pattern in BAD_STATUS_PATTERNS)), "default_target"] = 1
    return out


def normalize_rejected(df):
    out = pd.DataFrame()
    out["loan_amount"] = pd.to_numeric(df.get("Amount Requested"), errors="coerce")
    out["annual_inc"] = np.nan
    out["fico_score"] = pd.to_numeric(df.get("Risk_Score"), errors="coerce")
    out["dti"] = df.get("Debt-To-Income Ratio", np.nan).apply(parse_percent)
    out["term"] = 36
    out["emp_length_years"] = df.get("Employment Length", "").apply(parse_emp_length)
    out["revol_bal"] = np.nan
    out["revol_util"] = np.nan
    out["total_acc"] = np.nan
    out["open_acc"] = np.nan
    out["delinq_2yrs"] = np.nan
    out["inq_last_6mths"] = np.nan
    out["pub_rec"] = np.nan
    out["credit_history_years"] = np.nan
    out["state"] = df.get("State", "Unknown").fillna("Unknown").astype(str)
    out["purpose"] = df.get("Loan Title", "other").fillna("other").astype(str).str.lower().str.slice(0, 40)
    out["home_ownership"] = "Unknown"
    out["verification_status"] = "Unknown"
    out["application_type"] = "Individual"
    return out


def build_preprocessor(numeric_features=NUMERIC_FEATURES, categorical_features=CATEGORICAL_FEATURES):
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_features),
            ("cat", categorical, categorical_features),
        ],
        remainder="drop",
    )


def build_classifier(
    max_iter=180,
    learning_rate=0.06,
    l2_regularization=0.05,
    numeric_features=NUMERIC_FEATURES,
    categorical_features=CATEGORICAL_FEATURES,
):
    return Pipeline(
        steps=[
            ("prep", build_preprocessor(numeric_features, categorical_features)),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=max_iter,
                    learning_rate=learning_rate,
                    l2_regularization=l2_regularization,
                    random_state=42,
                ),
            ),
        ]
    )


def build_regressor(max_iter=220, learning_rate=0.05, l2_regularization=0.03):
    return Pipeline(
        steps=[
            ("prep", build_preprocessor()),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=max_iter,
                    learning_rate=learning_rate,
                    l2_regularization=l2_regularization,
                    random_state=42,
                ),
            ),
        ]
    )


def fit_binary_model(
    name,
    df,
    target,
    features=ALL_FEATURES,
    numeric_features=NUMERIC_FEATURES,
    categorical_features=CATEGORICAL_FEATURES,
):
    train_df = df.dropna(subset=[target]).copy()
    X_train, X_test, y_train, y_test = train_test_split(
        train_df[features],
        train_df[target].astype(int),
        test_size=0.22,
        random_state=42,
        stratify=train_df[target].astype(int),
    )
    model = build_classifier(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    metrics = {
        "rows": int(len(train_df)),
        "positive_rate": round(float(train_df[target].mean()), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
    }
    return model, metrics


def fit_interest_model(df):
    train_df = df.dropna(subset=["interest_rate", "loan_amount", "fico_score"]).copy()
    train_df = train_df[(train_df["interest_rate"] >= 3) & (train_df["interest_rate"] <= 35)]
    X_train, X_test, y_train, y_test = train_test_split(
        train_df[ALL_FEATURES],
        train_df["interest_rate"],
        test_size=0.22,
        random_state=42,
    )
    model = build_regressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "rows": int(len(train_df)),
        "mae": round(float(mean_absolute_error(y_test, preds)), 3),
        "rmse": round(float(mean_squared_error(y_test, preds) ** 0.5), 3),
        "r2": round(float(r2_score(y_test, preds)), 4),
    }
    return model, metrics


def train_models(sample_accepted=120_000, sample_rejected=120_000):
    accepted_path, rejected_path = find_lendingclub_files()
    accepted_cols = [
        "loan_amnt",
        "term",
        "int_rate",
        "annual_inc",
        "dti",
        "delinq_2yrs",
        "earliest_cr_line",
        "fico_range_low",
        "fico_range_high",
        "inq_last_6mths",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
        "home_ownership",
        "verification_status",
        "purpose",
        "addr_state",
        "application_type",
        "emp_length",
        "loan_status",
    ]
    rejected_cols = [
        "Amount Requested",
        "Loan Title",
        "Risk_Score",
        "Debt-To-Income Ratio",
        "State",
        "Employment Length",
    ]

    accepted_raw = read_sample(accepted_path, accepted_cols, sample_accepted, seed=42)
    rejected_raw = read_sample(rejected_path, rejected_cols, sample_rejected, seed=43)
    accepted = normalize_accepted(accepted_raw).dropna(subset=["loan_amount", "fico_score", "dti"])
    rejected = normalize_rejected(rejected_raw).dropna(subset=["loan_amount", "fico_score", "dti"])

    approval_accepted = accepted.copy()
    approval_accepted["accepted"] = 1
    approval_rejected = rejected.copy()
    approval_rejected["accepted"] = 0
    approval_df = pd.concat([approval_accepted, approval_rejected], ignore_index=True)

    approval_model, approval_metrics = fit_binary_model(
        "approval",
        approval_df,
        "accepted",
        features=APPROVAL_FEATURES,
        numeric_features=APPROVAL_NUMERIC_FEATURES,
        categorical_features=APPROVAL_CATEGORICAL_FEATURES,
    )
    default_model, default_metrics = fit_binary_model("default", accepted, "default_target")
    interest_model, interest_metrics = fit_interest_model(accepted)

    MODELS_DIR.mkdir(exist_ok=True)
    artifacts = {
        "version": "2026-04-29",
        "features": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "approval_model": approval_model,
        "default_model": default_model,
        "interest_model": interest_model,
    }
    joblib.dump(artifacts, ARTIFACT_PATH)

    model_card = {
        "project": "LendMatch",
        "data": {
            "accepted_csv": str(accepted_path.relative_to(PROJECT_ROOT)),
            "rejected_csv": str(rejected_path.relative_to(PROJECT_ROOT)),
            "accepted_sample_rows": int(len(accepted)),
            "rejected_sample_rows": int(len(rejected)),
        },
        "metrics": {
            "approval": approval_metrics,
            "default": default_metrics,
            "interest_rate": interest_metrics,
        },
        "notes": [
            "Models are trained from LendingClub historical accepted and rejected applications.",
            "Predictions are decision-support estimates, not a regulated credit decision engine.",
            "Charity deployments should add human review, fairness monitoring, and local compliance checks.",
        ],
    }
    MODEL_CARD_PATH.write_text(json.dumps(model_card, indent=2), encoding="utf-8")
    return model_card


@dataclass
class LendMatchPredictor:
    artifacts_path: Path = ARTIFACT_PATH

    def __post_init__(self):
        if not self.artifacts_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.artifacts_path}. Run `python -m src.lendmatch_model` first."
            )
        self.artifacts = joblib.load(self.artifacts_path)
        self.approval_model = self.artifacts["approval_model"]
        self.default_model = self.artifacts["default_model"]
        self.interest_model = self.artifacts["interest_model"]

    def normalize_application(self, data):
        fico = clamp(data.get("fico_score"), 300, 850, 680)
        income = clamp(data.get("annual_inc"), 1_000, 5_000_000, 55_000)
        amount = clamp(data.get("loan_amount"), 500, 100_000, 10_000)
        dti = clamp(data.get("dti"), 0, 80, 20)
        term = int(clamp(data.get("term"), 12, 84, 36))
        emp_value = data.get("emp_length", data.get("emp_length_years", 2))

        return pd.DataFrame(
            [
                {
                    "loan_amount": amount,
                    "annual_inc": income,
                    "fico_score": fico,
                    "dti": dti,
                    "term": term,
                    "emp_length_years": parse_emp_length(emp_value),
                    "revol_bal": clamp(data.get("revol_bal"), 0, 1_000_000, max(0, income * 0.15)),
                    "revol_util": clamp(data.get("revol_util"), 0, 150, 35),
                    "total_acc": clamp(data.get("total_acc"), 0, 200, 18),
                    "open_acc": clamp(data.get("open_acc"), 0, 100, 8),
                    "delinq_2yrs": clamp(data.get("delinq_2yrs"), 0, 50, 0),
                    "inq_last_6mths": clamp(data.get("inq_last_6mths"), 0, 25, 0),
                    "pub_rec": clamp(data.get("pub_rec"), 0, 25, 0),
                    "credit_history_years": clamp(data.get("credit_history_years"), 0, 70, 8),
                    "state": str(data.get("state", "CA")).upper()[:2],
                    "purpose": str(data.get("purpose", "debt_consolidation")),
                    "home_ownership": str(data.get("home_ownership", "RENT")).upper(),
                    "verification_status": str(data.get("verification_status", "Source Verified")),
                    "application_type": str(data.get("application_type", "Individual")),
                }
            ]
        )

    def predict(self, data):
        X = self.normalize_application(data)
        approval_probability = float(self.approval_model.predict_proba(X)[0, 1])
        default_probability = float(self.default_model.predict_proba(X)[0, 1])
        interest_rate = float(self.interest_model.predict(X)[0])
        interest_rate = round(max(4.0, min(34.99, interest_rate)), 2)

        risk_band = "Low"
        if default_probability >= 0.18:
            risk_band = "High"
        elif default_probability >= 0.09:
            risk_band = "Moderate"

        decision = "Refer"
        if approval_probability >= 0.68 and default_probability < 0.16:
            decision = "Eligible"
        elif approval_probability < 0.38 or default_probability >= 0.24:
            decision = "Needs support"

        return {
            "decision": decision,
            "approval_probability": approval_probability,
            "default_probability": default_probability,
            "predicted_interest_rate": interest_rate,
            "risk_band": risk_band,
            "normalized_application": X.iloc[0].to_dict(),
        }


if __name__ == "__main__":
    accepted = int(os.environ.get("LENDMATCH_ACCEPTED_SAMPLE", "120000"))
    rejected = int(os.environ.get("LENDMATCH_REJECTED_SAMPLE", "120000"))
    card = train_models(sample_accepted=accepted, sample_rejected=rejected)
    print(json.dumps(card["metrics"], indent=2))
