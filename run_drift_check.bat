@echo off
echo Running drift detection and retraining pipeline (if needed)...
call venv\Scripts\activate
python check_drift_RFM.py
pause
