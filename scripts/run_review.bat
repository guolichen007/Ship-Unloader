@echo off
setlocal
set "ROOT=%~dp0.."
set "OPENBLAS_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
set "PYTHONNOUSERSITE=1"
set "PYTHONIOENCODING=utf-8"
set "MPLBACKEND=Agg"
if not defined PYTHON_BIN set "PYTHON_BIN=%ROOT%\.local\review-venv\Scripts\python.exe"
"%PYTHON_BIN%" -s "%ROOT%\tools\r3h_oneclick.py" %*
exit /b %errorlevel%
