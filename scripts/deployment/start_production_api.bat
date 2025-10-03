@echo off
REM Start Production API Server
REM Uses ml-conda environment automatically

echo ========================================
echo Starting Production API Server
echo Environment: ml-conda
echo ========================================
echo.

REM Activate ml-conda and start server
call conda activate ml-conda
if errorlevel 1 (
    echo ERROR: Failed to activate ml-conda environment
    echo Please ensure conda is initialized and ml-conda environment exists
    pause
    exit /b 1
)

echo Environment activated: ml-conda
echo Starting API server...
echo.

REM Change to project root directory
cd /d "%~dp0..\.."

python scripts\deployment\start_api.py --production --port 5000

pause
