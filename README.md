# HR Analytics – Advanced Employee Attrition Prediction

## Project objective
Build an end-to-end, explainable HR analytics system that:
1. profiles and cleans employee data,
2. performs exploratory and statistical analysis,
3. engineers HR-specific features,
4. trains and compares multiple classifiers,
5. handles class imbalance,
6. tunes the decision threshold,
7. explains predictions with SHAP,
8. audits model performance across selected employee groups,
9. exports Power BI-ready data,
10. provides an interactive Streamlit prediction dashboard.

## Open-source dataset
This project is designed for the public **IBM HR Analytics Employee Attrition & Performance** dataset
(commonly distributed as `WA_Fn-UseC_-HR-Employee-Attrition.csv`, 1,470 rows × 35 columns).

Primary public reference:
- Kaggle dataset page: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
- Public GitHub mirror used by the downloader:
  https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv

The repository intentionally does not claim the model discovers causal relationships. SHAP describes model behavior; it does not prove that a feature causes resignation.

## Folder structure
```text
HR_Attrition_Advanced_v2/
├── app/streamlit_app.py
├── data/                     # downloaded dataset goes here
├── models/                   # trained model artifacts
├── notebooks/                # notebook-ready workflow notes
├── outputs/                  # metrics, predictions, charts, Power BI CSV
├── powerbi/PowerBI_Guide.md
├── reports/Project_Report.md
├── src/
│   ├── download_data.py
│   ├── pipeline.py
│   ├── shap_analysis.py
│   └── fairness_audit.py
├── requirements.txt
└── README.md
```

## Quick start

### 1. Create environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the open dataset
```bash
python src/download_data.py
```

### 4. Run the complete ML pipeline
```bash
python src/pipeline.py
```

### 5. Run SHAP analysis
```bash
python src/shap_analysis.py
```

### 6. Run the group/fairness audit
```bash
python src/fairness_audit.py
```

### 7. Launch the interactive application
```bash
streamlit run app/streamlit_app.py
```

## Main outputs
After execution, check:
- `outputs/metrics.csv`
- `outputs/classification_report.txt`
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`
- `outputs/pr_curve.png`
- `outputs/feature_importance.png`
- `outputs/predictions.csv`
- `outputs/powerbi_employee_attrition.csv`
- `outputs/shap_summary.png`
- `outputs/group_audit.csv`
- `models/attrition_model.joblib`

## Advanced features
- Stratified train/test split
- Preprocessing pipeline with one-hot encoding and scaling
- Median imputation
- Logistic Regression, Random Forest and HistGradientBoosting
- Class weighting where supported
- 5-fold cross-validation
- ROC-AUC and PR-AUC
- Threshold optimization using F1 on validation data
- Calibration curve
- SHAP explainability when SHAP is installed
- Group performance audit by gender, age band and overtime
- Reproducible random seed
- Power BI export
- Streamlit what-if prediction interface

## Academic note
Use the generated metrics from your own run in the final report. Do not invent accuracy values. Because this dataset is imbalanced, report precision, recall, F1 and PR-AUC alongside accuracy.
