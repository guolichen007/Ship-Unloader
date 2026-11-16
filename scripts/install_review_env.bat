@echo off
setlocal
set "ROOT=%~dp0.."
if defined PYTHON_BIN (
  "%PYTHON_BIN%" -c "import sys; assert (3,10) <= sys.version_info < (3,13), 'PYTHON_VERSION_UNSUPPORTED: use Python 3.10 to 3.12.'"
  if errorlevel 1 exit /b 1
  "%PYTHON_BIN%" -m venv "%ROOT%\.local\review-venv"
) else (
  py -3.10 -m venv "%ROOT%\.local\review-venv"
)
if errorlevel 1 exit /b 1
"%ROOT%\.local\review-venv\Scripts\python.exe" -m pip install -r "%ROOT%\requirements-ros.txt"
if errorlevel 1 exit /b 1
"%ROOT%\.local\review-venv\Scripts\python.exe" -c "import numpy, scipy, matplotlib, jsonschema, yaml; print('REVIEW_ENVIRONMENT_READY')"
exit /b %errorlevel%
