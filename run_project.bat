@echo off
python src\download_data.py
python src\pipeline.py
python src\shap_analysis.py
python src\fairness_audit.py
echo.
echo Project complete. Check the outputs folder.
pause
