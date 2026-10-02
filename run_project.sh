#!/usr/bin/env bash
set -e
python src/download_data.py
python src/pipeline.py
python src/shap_analysis.py
python src/fairness_audit.py
echo "Project complete. Check the outputs folder."
