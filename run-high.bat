@echo off
setlocal EnableExtensions
pushd "%~dp0"

set "PYTHON=.\venv\Scripts\python.exe"
set "inference_memory=2560"
set "DML_DEVICE=0"

rem --- Perfil HIGH: intenta FP16 en UNet/CLIP, normal_vram, atención SPLIT ---
rem Nota: quitamos --no-half; si falla, volvé a MEDIUM.
set "ARGS=--directml %DML_DEVICE% --skip-install --skip-torch-cuda-test --precision full"
set "ARGS=%ARGS% --always-normal-vram --vae-in-cpu --attention-split --unet-in-fp16 --clip-in-fp16"

echo ===============================================
echo Perfil:   HIGH (experimental)
echo DML dev:  %DML_DEVICE%
echo inf_mem:  %inference_memory% MB
echo Args:     %ARGS%
echo ===============================================

"%PYTHON%" launch.py %ARGS%
if errorlevel 1 pause

popd
endlocal
