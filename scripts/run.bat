@echo off
title 🚀 Animal Classifier - Streamlit + Training + Reports

echo ===============================
echo 🟢 Launching Streamlit App...
echo ===============================
start cmd /k "cd /d C:\Users\avish\Documents\animal_classifier_unified_mentor\app && streamlit run app.py"

echo ===============================
echo 🔄 Training Model in Background...
echo ===============================
start cmd /c "cd /d C:\Users\avish\Documents\animal_classifier_unified_mentor\src\training && python train.py"

timeout /t 5 >nul

echo ===============================
echo 📊 Generating Reports...
echo ===============================
start cmd /c "cd /d C:\Users\avish\Documents\animal_classifier_unified_mentor && python scripts\generate_report.py"

echo.
echo ✅ Streamlit is running. Training and report generation have started in background.
pause
