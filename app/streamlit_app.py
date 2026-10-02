from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "models" / "attrition_model.joblib"

st.set_page_config(page_title="HR Attrition Predictor", page_icon="👥", layout="wide")
st.title("👥 HR Employee Attrition Prediction")
st.caption("Explainable machine-learning demonstration based on the public IBM HR Analytics dataset.")

if not ARTIFACT.exists():
    st.error("Model artifact not found. Run `python src/pipeline.py` first.")
    st.stop()

artifact = joblib.load(ARTIFACT)
model = artifact["model"]
feature_columns = artifact["feature_columns"]
selected_model = artifact["selected_model"]

st.info(f"Loaded model: {selected_model}")

# Build a robust form using common IBM HR fields.
defaults = {
    "Age": 34, "DailyRate": 800, "DistanceFromHome": 5,
    "Education": 3, "EmployeeCount": 1, "EnvironmentSatisfaction": 3,
    "HourlyRate": 65, "JobInvolvement": 3, "JobLevel": 2,
    "JobSatisfaction": 3, "MonthlyIncome": 5000, "MonthlyRate": 15000,
    "NumCompaniesWorked": 2, "PercentSalaryHike": 14,
    "PerformanceRating": 3, "RelationshipSatisfaction": 3,
    "StandardHours": 80, "StockOptionLevel": 1, "TotalWorkingYears": 8,
    "TrainingTimesLastYear": 3, "WorkLifeBalance": 3,
    "YearsAtCompany": 5, "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 3,
}

categorical = {
    "BusinessTravel": ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
    "Department": ["Sales", "Research & Development", "Human Resources"],
    "EducationField": ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
    "Gender": ["Male", "Female"],
    "JobRole": ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
                "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"],
    "MaritalStatus": ["Single", "Married", "Divorced"],
    "OverTime": ["Yes", "No"],
}

st.subheader("Employee profile")
inputs = {}
cols = st.columns(3)

for i, col in enumerate(feature_columns):
    if col in categorical:
        inputs[col] = cols[i % 3].selectbox(col, categorical[col])
    elif col in defaults:
        inputs[col] = cols[i % 3].number_input(col, value=float(defaults[col]), step=1.0)
    elif col in ["AgeBand", "IncomeBand", "TenureBand"]:
        # These are engineered in the pipeline and should be derived below.
        continue
    elif col in ["RoleTenureRatio", "PromotionGapRatio", "SatisfactionIndex", "EarlyCareer"]:
        continue
    else:
        # Unknown feature: safe numeric fallback.
        inputs[col] = cols[i % 3].number_input(col, value=0.0, step=1.0)

row = pd.DataFrame([inputs])

# Engineer the same features as the training pipeline.
if "MonthlyIncome" in row:
    row["IncomeBand"] = pd.cut(row["MonthlyIncome"], [-np.inf,3000,6000,10000,np.inf],
                               labels=["Low","Medium","High","Very High"])
if "Age" in row:
    row["AgeBand"] = pd.cut(row["Age"], [-np.inf,25,35,45,55,np.inf],
                            labels=["<=25","26-35","36-45","46-55","56+"])
if "YearsAtCompany" in row:
    row["TenureBand"] = pd.cut(row["YearsAtCompany"], [-np.inf,1,3,7,12,np.inf],
                               labels=["0-1","2-3","4-7","8-12","13+"])
if {"YearsAtCompany","YearsInCurrentRole"}.issubset(row.columns):
    row["RoleTenureRatio"] = (row["YearsInCurrentRole"] /
                              row["YearsAtCompany"].replace(0, np.nan)).fillna(0).clip(0,1)
if {"YearsSinceLastPromotion","YearsAtCompany"}.issubset(row.columns):
    row["PromotionGapRatio"] = (row["YearsSinceLastPromotion"] /
                                row["YearsAtCompany"].replace(0, np.nan)).fillna(0).clip(0,1)
sat = [c for c in ["JobSatisfaction","EnvironmentSatisfaction","RelationshipSatisfaction","WorkLifeBalance"] if c in row]
if sat:
    row["SatisfactionIndex"] = row[sat].mean(axis=1)
if {"Age","YearsAtCompany"}.issubset(row.columns):
    row["EarlyCareer"] = ((row["Age"] <= 30) & (row["YearsAtCompany"] <= 3)).astype(int)

# Remove constants and align columns.
row = row.drop(columns=[c for c in ["EmployeeCount","StandardHours","Over18","EmployeeNumber"] if c in row], errors="ignore")
for c in feature_columns:
    if c not in row:
        row[c] = 0
row = row[feature_columns]

if st.button("Predict Attrition Risk", type="primary"):
    probability = float(model.predict_proba(row)[:,1][0])
    st.metric("Predicted attrition probability", f"{probability:.1%}")
    if probability >= 0.50:
        st.warning("The model assigns this profile a higher predicted risk at a 0.50 threshold.")
    else:
        st.success("The model assigns this profile a lower predicted risk at a 0.50 threshold.")
    st.caption("This is a statistical prediction, not a certainty or a recommendation to take action against an employee.")
