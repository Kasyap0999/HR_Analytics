from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "HR_Employee_Attrition.csv"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

RANDOM_STATE = 42

def load_data():
    if not DATA.exists():
        raise FileNotFoundError("Dataset not found. Run: python src/download_data.py")
    df = pd.read_csv(DATA)
    # Normalize common target spelling.
    if "Attrition" not in df.columns:
        candidates = [c for c in df.columns if c.lower() == "attrition"]
        if candidates:
            df = df.rename(columns={candidates[0]: "Attrition"})
    if "Attrition" not in df.columns:
        raise ValueError("Target column 'Attrition' was not found.")
    return df

def feature_engineering(df):
    df = df.copy()
    if "Attrition" in df:
        df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0}).fillna(df["Attrition"])

    if "MonthlyIncome" in df:
        df["IncomeBand"] = pd.cut(
            df["MonthlyIncome"], [-np.inf, 3000, 6000, 10000, np.inf],
            labels=["Low", "Medium", "High", "Very High"]
        )
    if "Age" in df:
        df["AgeBand"] = pd.cut(
            df["Age"], [-np.inf, 25, 35, 45, 55, np.inf],
            labels=["<=25", "26-35", "36-45", "46-55", "56+"]
        )
    if "YearsAtCompany" in df:
        df["TenureBand"] = pd.cut(
            df["YearsAtCompany"], [-np.inf, 1, 3, 7, 12, np.inf],
            labels=["0-1", "2-3", "4-7", "8-12", "13+"]
        )
    if {"YearsAtCompany", "YearsInCurrentRole"}.issubset(df.columns):
        df["RoleTenureRatio"] = df["YearsInCurrentRole"] / df["YearsAtCompany"].replace(0, np.nan)
        df["RoleTenureRatio"] = df["RoleTenureRatio"].fillna(0).clip(0, 1)
    if {"YearsSinceLastPromotion", "YearsAtCompany"}.issubset(df.columns):
        df["PromotionGapRatio"] = (
            df["YearsSinceLastPromotion"] / df["YearsAtCompany"].replace(0, np.nan)
        ).fillna(0).clip(0, 1)
    satisfaction_cols = [
        c for c in ["JobSatisfaction", "EnvironmentSatisfaction",
                    "RelationshipSatisfaction", "WorkLifeBalance"]
        if c in df.columns
    ]
    if satisfaction_cols:
        df["SatisfactionIndex"] = df[satisfaction_cols].mean(axis=1)
    if {"Age", "YearsAtCompany"}.issubset(df.columns):
        df["EarlyCareer"] = ((df["Age"] <= 30) & (df["YearsAtCompany"] <= 3)).astype(int)

    # Remove constant/identifier columns commonly present in the IBM dataset.
    drop_cols = [
        "Attrition", "EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"
    ]
    return df, [c for c in drop_cols if c in df.columns]

def make_preprocessor(X):
    cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num = X.select_dtypes(include=[np.number]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", num_pipe, num),
        ("cat", cat_pipe, cat)
    ])

def evaluate(name, estimator, Xtr, Xte, ytr, yte):
    estimator.fit(Xtr, ytr)
    prob = estimator.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(yte, pred),
        "precision": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred, zero_division=0),
        "f1": f1_score(yte, pred, zero_division=0),
        "roc_auc": roc_auc_score(yte, prob),
        "pr_auc": average_precision_score(yte, prob)
    }
    return estimator, prob, pred, metrics

def main():
    df = load_data()
    df, drop_cols = feature_engineering(df)
    y = df["Attrition"].astype(int)
    X = df.drop(columns=drop_cols, errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    pre = make_preprocessor(X_train)
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=RANDOM_STATE
        )
    }

    results, fitted = [], {}
    for name, model in models.items():
        pipe = Pipeline([("preprocessor", pre), ("model", model)])
        pipe, prob, pred, metrics = evaluate(
            name, pipe, X_train, X_test, y_train, y_test
        )
        fitted[name] = (pipe, prob, pred)
        results.append(metrics)

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(OUT / "metrics.csv", index=False)

    # Cross-validation for each model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []
    for name, model in models.items():
        pipe = Pipeline([("preprocessor", make_preprocessor(X_train)), ("model", model)])
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            n_jobs=-1
        )
        cv_rows.append({
            "model": name,
            **{f"cv_{k}_mean": float(np.mean(v))
               for k, v in [(m, scores[f"test_{m}"]) for m in ["accuracy","precision","recall","f1","roc_auc"]]},
            **{f"cv_{k}_std": float(np.std(v))
               for k, v in [(m, scores[f"test_{m}"]) for m in ["accuracy","precision","recall","f1","roc_auc"]]}
        })
    pd.DataFrame(cv_rows).to_csv(OUT / "cross_validation.csv", index=False)

    # Select model by validation-free ROC-AUC only for artifact creation; show all metrics in report.
    selected_name = metrics_df.sort_values("roc_auc", ascending=False).iloc[0]["model"]
    selected_pipe, selected_prob, selected_pred = fitted[selected_name]
    joblib.dump(
        {"model": selected_pipe, "feature_columns": X.columns.tolist(),
         "selected_model": selected_name},
        MODELS / "attrition_model.joblib"
    )

    # Classification report
    with open(OUT / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Selected model: {selected_name}\n\n")
        f.write(classification_report(y_test, selected_pred, digits=4))
        f.write("\nNote: threshold=0.50 for this report.\n")

    # Confusion matrix
    cm = confusion_matrix(y_test, selected_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", square=True, cbar=False)
    plt.title(f"Confusion Matrix – {selected_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=180)
    plt.close()

    # ROC
    plt.figure(figsize=(7, 5))
    for name, (_, prob, _) in fitted.items():
        RocCurveDisplay.from_predictions(y_test, prob, name=name)
    plt.tight_layout()
    plt.savefig(OUT / "roc_curve.png", dpi=180)
    plt.close()

    # PR curve
    plt.figure(figsize=(7, 5))
    for name, (_, prob, _) in fitted.items():
        PrecisionRecallDisplay.from_predictions(y_test, prob, name=name)
    plt.tight_layout()
    plt.savefig(OUT / "pr_curve.png", dpi=180)
    plt.close()

    # Threshold analysis for selected model
    thresholds = np.arange(0.10, 0.91, 0.05)
    threshold_rows = []
    for t in thresholds:
        p = (selected_prob >= t).astype(int)
        threshold_rows.append({
            "threshold": round(float(t), 2),
            "precision": precision_score(y_test, p, zero_division=0),
            "recall": recall_score(y_test, p, zero_division=0),
            "f1": f1_score(y_test, p, zero_division=0)
        })
    th = pd.DataFrame(threshold_rows)
    th.to_csv(OUT / "threshold_analysis.csv", index=False)
    best_t = float(th.loc[th["f1"].idxmax(), "threshold"])
    pd.DataFrame([{"selected_model": selected_name, "recommended_demo_threshold": best_t}]).to_csv(
        OUT / "threshold_summary.csv", index=False
    )

    # Feature importance using permutation importance on raw features.
    pi = permutation_importance(
        selected_pipe, X_test, y_test, scoring="roc_auc",
        n_repeats=8, random_state=RANDOM_STATE, n_jobs=-1
    )
    imp = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std
    }).sort_values("importance_mean", ascending=False)
    imp.to_csv(OUT / "feature_importance.csv", index=False)

    top = imp.head(15).sort_values("importance_mean")
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance_mean"])
    plt.xlabel("Mean decrease in ROC-AUC")
    plt.title("Permutation Feature Importance")
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance.png", dpi=180)
    plt.close()

    # Power BI export
    export = df.copy()
    export["PredictedAttritionProbability"] = selected_pipe.predict_proba(X)[:, 1]
    export["PredictedAttrition"] = np.where(
        export["PredictedAttritionProbability"] >= best_t, "Yes", "No"
    )
    export.to_csv(OUT / "powerbi_employee_attrition.csv", index=False)
    export.to_csv(OUT / "predictions.csv", index=False)

    # Data quality summary
    quality = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing": [int(df[c].isna().sum()) for c in df.columns],
        "unique": [int(df[c].nunique(dropna=True)) for c in df.columns]
    })
    quality.to_csv(OUT / "data_quality.csv", index=False)

    print("Pipeline complete.")
    print(metrics_df.to_string(index=False))
    print(f"Selected model: {selected_name}")
    print(f"Demo threshold maximizing test-set F1: {best_t:.2f}")
    print(f"Outputs saved to: {OUT}")

if __name__ == "__main__":
    main()
