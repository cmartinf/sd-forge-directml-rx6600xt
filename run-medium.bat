@echo off
setlocal EnableExtensions
pushd "%~dp0"

set "PYTHON=.\venv\Scripts\python.exe"
set "inference_memory=2048"
set "DML_DEVICE=0"

rem MEDIUM: FP16, low_vram, sub-quadratic con chunks moderados, VAE en CPU
set "ARGS=--directml %DML_DEVICE% --skip-install --skip-torch-cuda-test --precision full"
set "ARGS=%ARGS% --always-low-vram --vae-in-cpu --opt-sub-quad-attention"
set "ARGS=%ARGS% --sub-quad-q-chunk-size 160 --sub-quad-kv-chunk-size 160 --sub-quad-chunk-threshold 0"
set "ARGS=%ARGS% --unet-in-fp16 --clip-in-fp16"

echo ===============================================
echo Perfil:   MEDIUM
echo DML dev:  %DML_DEVICE%
echo inf_mem:  %inference_memory% MB
echo Args:     %ARGS%
echo ===============================================

"%PYTHON%" launch.py %ARGS%
if errorlevel 1 pause
popd
endlocal

