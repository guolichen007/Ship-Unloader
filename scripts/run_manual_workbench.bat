@echo off
setlocal
set "ROOT=%~dp0.."
if not defined PYTHON_BIN set "PYTHON_BIN=%ROOT%\.local\review-venv\Scripts\python.exe"
set "PYTHONPATH=%ROOT%;%PYTHONPATH%"
set "PYTHONNOUSERSITE=1"
set "OPENBLAS_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
"%PYTHON_BIN%" -s -m ship_perception.manual_review.workbench_server %*
exit /b %errorlevel%
