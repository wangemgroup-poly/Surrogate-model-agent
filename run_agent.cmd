@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set OPENBLAS_NUM_THREADS=4
set OMP_NUM_THREADS=4
if not exist ".venv\Scripts\python.exe" (
  echo Please run setup.cmd first. Python 3.12 is required for CST 2025.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" agent.py %*
pause
