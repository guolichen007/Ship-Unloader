@echo off
setlocal
set "OPENBLAS_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
if not defined PYTHON_BIN set "PYTHON_BIN=python"
"%PYTHON_BIN%" "%~dp0..\tools\r3h_oneclick.py" %*
exit /b %errorlevel%
