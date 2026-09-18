@echo off
cd /d "%~dp0"
if "%~1"=="" (py -3.12 -m venv .venv) else ("%~1" -m venv .venv)
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
rem SB-SADEA BNN (Liu et al., IEEE TAP 2022) uses PyTorch; CUDA 12.6 build, falls back to CPU when no GPU.
".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 exit /b 1
echo Setup complete. Open the Agent launcher.
pause
