from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "HR_Employee_Attrition.csv"
OUT = ROOT / "outputs"

def metric_row(group, y, p, prob):
    return {
        "group": group,
        "n": len(y),
        "actual_attrition_rate": float(np.mean(y)),
        "predicted_positive_rate": float(np.mean(p)),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "f1": f1_score(y, p, zero_division=0),
        "roc_auc": roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan
    }

def main():
    pred_file = OUT / "predictions.csv"
    if not pred_file.exists():
        raise SystemExit("Run python src/pipeline.py first.")

    df = pd.read_csv(pred_file)
    y = df["Attrition"].astype(int)
    prob = df["PredictedAttritionProbability"].astype(float)
    p = (df["PredictedAttrition"] == "Yes").astype(int)

    rows = []
    for col in ["Gender", "AgeBand", "OverTime"]:
        if col in df.columns:
            for value, idx in df.groupby(col).groups.items():
                rows.append(metric_row(f"{col}={value}", y.loc[idx], p.loc[idx], prob.loc[idx]))

    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "group_audit.csv", index=False)
    print(audit.to_string(index=False))
    print("\nThis is a descriptive performance audit, not a legal fairness certification.")

if __name__ == "__main__":
    main()
