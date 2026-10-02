from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "HR_Employee_Attrition.csv"
OUT = ROOT / "outputs"

def main():
    try:
        import shap
    except ImportError:
        raise SystemExit("Install SHAP first: pip install shap")

    artifact = joblib.load(ROOT / "models" / "attrition_model.joblib")
    pipe = artifact["model"]
    df = pd.read_csv(DATA)

    # Match feature engineering used by pipeline.py
    from pipeline import feature_engineering
    df, drop_cols = feature_engineering(df)
    X = df.drop(columns=drop_cols, errors="ignore")

    sample = X.sample(min(400, len(X)), random_state=42)
    pre = pipe.named_steps["preprocessor"]
    model = pipe.named_steps["model"]
    transformed = pre.transform(sample)
    names = pre.get_feature_names_out()

    # TreeExplainer for RF; LinearExplainer for logistic; generic fallback otherwise.
    if model.__class__.__name__ == "RandomForestClassifier":
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[1]
    elif model.__class__.__name__ == "LogisticRegression":
        explainer = shap.LinearExplainer(model, transformed)
        values = explainer.shap_values(transformed)
    else:
        # Kernel/Exact can be expensive; use a small sample.
        background = transformed[:50]
        explain_sample = transformed[:100]
        explainer = shap.Explainer(model.predict_proba, background)
        result = explainer(explain_sample)
        values = result.values
        if values.ndim == 3:
            values = values[:, :, 1]

    plt.figure()
    shap.summary_plot(values, transformed, feature_names=names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()

    importance = np.abs(values).mean(axis=0)
    shap_df = pd.DataFrame({"feature": names, "mean_abs_shap": importance})
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(OUT / "shap_importance.csv", index=False)

    print("SHAP analysis complete.")
    print(shap_df.head(15).to_string(index=False))

if __name__ == "__main__":
    main()
