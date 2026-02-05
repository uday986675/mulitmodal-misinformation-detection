@echo off
REM Run Streamlit App - Windows Version
REM Simple batch file to start the Streamlit application

cls
echo ================================
echo 🚀 Starting Misinformation Detector
echo ================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8 or higher.
    echo    Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo ✅ Python found: %PYTHON_VERSION%
echo.

REM Check if Streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo 📦 Streamlit not found. Installing...
    pip install streamlit==1.28.0
    echo.
)

REM Check if model checkpoint exists
if not exist "checkpoints\final_model.pt" (
    echo ⚠️  Warning: Model checkpoint not found at checkpoints\final_model.pt
    echo.
)

echo 🔄 Starting Streamlit app...
echo.
echo The app will open at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ================================
echo.

REM Run Streamlit app
python -m streamlit run app.py

pause
