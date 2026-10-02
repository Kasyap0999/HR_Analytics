# HR Analytics – Advanced Employee Attrition Prediction

## 1. Abstract
This project develops an end-to-end HR analytics workflow using the public IBM HR Analytics Employee Attrition & Performance dataset. The workflow combines exploratory data analysis, feature engineering, supervised classification, threshold analysis, SHAP explainability, descriptive group auditing and Power BI visualization.

## 2. Problem statement
Employee attrition can affect recruitment workload, training costs and organizational continuity. The project studies patterns associated with observed attrition and builds a model that estimates the probability of attrition for an employee profile.

## 3. Dataset
Use the open dataset described in `README.md`. Record the actual row/column counts after download.

## 4. Data preparation
- Missing-value profiling
- Duplicate check
- Identifier/constant-column removal
- Numeric scaling
- Categorical one-hot encoding
- Stratified train/test split

## 5. Feature engineering
- AgeBand
- IncomeBand
- TenureBand
- RoleTenureRatio
- PromotionGapRatio
- SatisfactionIndex
- EarlyCareer

## 6. Exploratory analysis
Include charts for:
- overall attrition,
- department,
- job role,
- overtime,
- salary band,
- satisfaction,
- tenure,
- promotion gap.

Interpret differences descriptively. Avoid causal claims.

## 7. Machine learning
Models:
- Logistic Regression
- Random Forest
- HistGradientBoosting

Evaluation:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- 5-fold cross-validation

## 8. Threshold analysis
The default 0.50 threshold is not always appropriate for imbalanced classification. The pipeline produces `threshold_analysis.csv` so precision/recall/F1 can be examined at multiple thresholds.

## 9. Explainability
SHAP is used to explain model outputs. Report the most influential transformed features and clearly distinguish model explanation from causation.

## 10. Group audit
`fairness_audit.py` produces descriptive performance metrics by selected groups such as gender, age band and overtime. Interpret these as model-performance diagnostics, not as a legal compliance certification.

## 11. Power BI
Build an interactive dashboard using the exported Power BI CSV and the guide in `powerbi/PowerBI_Guide.md`.

## 12. Recommendations
Recommendations should be connected to observed patterns. Examples:
- review workload when overtime is strongly associated with predicted risk,
- review career development pathways when promotion gaps are prominent,
- examine compensation and internal mobility where salary/role patterns are substantial,
- use retention interviews and manager feedback to validate quantitative patterns.

## 13. Conclusion
The project demonstrates an explainable, reproducible approach to HR attrition analytics. The final system combines descriptive analytics, predictive modeling and interactive reporting while avoiding unsupported causal conclusions.

## 14. Limitations
- Dataset is historical and synthetic/fictional in nature as described by its public documentation.
- Model predictions are probabilistic.
- Observational patterns do not establish causality.
- Group metrics depend on sample sizes and should be interpreted carefully.
