@echo off
setlocal EnableExtensions
pushd "%~dp0"

rem --- Python del venv ---
set "PYTHON=.\venv\Scripts\python.exe"

rem === GPU DirectML: 0 para tu RX 6600 XT (vacío = default) ===
set "DML_DEVICE=0"

rem === Memoria reservada para softmax/atención (MB) ===
set "INFERENCE_MEMORY=2048"
rem Forge mira la var en minúsculas:
set "inference_memory=%INFERENCE_MEMORY%"

rem === LIMPIAR var heredada de Forge para evitar duplicados ===
set "COMMANDLINE_ARGS="

echo ===============================================
echo Perfil primario:  FORCE_SPLIT
echo DirectML device:  %DML_DEVICE%
echo inference_memory: %INFERENCE_MEMORY% MB
echo ===============================================

rem --------- 1) INTENTO: forzar cross-attention SPLIT ---------
set "ARGS=--directml"
if defined DML_DEVICE set "ARGS=%ARGS% %DML_DEVICE%"
set "ARGS=%ARGS% --skip-install --skip-torch-cuda-test --precision full --no-half"
set "ARGS=%ARGS% --always-low-vram --vae-in-cpu"
rem Forzamos el backend moderno de atención en modo split:
set "ARGS=%ARGS% --attention-split"

echo Args: %ARGS%
"%PYTHON%" launch.py %ARGS%
IF %ERRORLEVEL% EQU 0 GOTO :EOF

echo.
echo [!] FORCE_SPLIT falló o se cerró con error. Probamos SUBQUAD_ULTRA...
echo.

rem --------- 2) BACKUP: sub-quadratic con troceo agresivo ---------
set "ARGS=--directml"
if defined DML_DEVICE set "ARGS=%ARGS% %DML_DEVICE%"
set "ARGS=%ARGS% --skip-install --skip-torch-cuda-test --precision full --no-half"
set "ARGS=%ARGS% --always-low-vram --vae-in-cpu"
set "ARGS=%ARGS% --opt-sub-quad-attention --sub-quad-q-chunk-size 128 --sub-quad-kv-chunk-size 160 --sub-quad-chunk-threshold 0"

echo Args: %ARGS%
"%PYTHON%" launch.py %ARGS%

popd
endlocal

