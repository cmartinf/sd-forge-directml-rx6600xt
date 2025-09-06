@echo off
setlocal EnableExtensions
pushd "%~dp0"

rem --- Python del venv ---
set "PYTHON=.\venv\Scripts\python.exe"

rem === Memoria reservada para softmax/atención (MB) ===
set "inference_memory=2048"

rem === Elegir GPU DirectML (0 por defecto) ===
set "DML_DEVICE=0"

rem === Perfil XL (768x768) ===
set "ARGS=--directml %DML_DEVICE% --skip-install --skip-torch-cuda-test --precision full"
rem FP16 en UNet/CLIP + ahorro de VRAM
set "ARGS=%ARGS% --unet-in-fp16 --clip-in-fp16 --always-low-vram --always-offload-from-vram --vae-in-cpu --opt-channelslast"
rem Sub-quadratic attention con chunks para alta resolución
set "ARGS=%ARGS% --opt-sub-quad-attention --sub-quad-q-chunk-size 112 --sub-quad-kv-chunk-size 112 --sub-quad-chunk-threshold 0"

echo ===============================================
echo Perfil:   XL (alta resolucion)
echo DML dev:  %DML_DEVICE%
echo inf_mem:  %inference_memory% MB
echo Args:     %ARGS%
echo ===============================================

"%PYTHON%" launch.py %ARGS%
if errorlevel 1 pause

popd
endlocal
