@echo off
setlocal
set "STREAMCLIP_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%STREAMCLIP_PYTHON%" (
    echo Missing virtual environment. Create .venv and install requirements.txt first.
    exit /b 2
)
"%STREAMCLIP_PYTHON%" "%~dp0scripts\run_pipeline.py" %*
exit /b %errorlevel%
