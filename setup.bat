@echo off
REM One-time setup. Needs: Python 3.12 and Node.js (LTS) on PATH, plus internet for the installs.
setlocal
cd /d "%~dp0"

echo [1/5] Main backend environment...
if not exist .venv ( python -m venv .venv || goto :fail )
call .venv\Scripts\python.exe -m pip install --upgrade pip || goto :fail
call .venv\Scripts\python.exe -m pip install -r requirements.txt || goto :fail

echo [2/5] Checking for an NVIDIA GPU...
where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
  echo   NVIDIA GPU detected - installing CUDA-enabled PyTorch instead of the CPU build ^(~2GB download^)...
  call .venv\Scripts\python.exe -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu130 --timeout 120 --retries 10 || goto :fail
) else (
  echo   No NVIDIA GPU detected - keeping CPU-only PyTorch. AI video analysis needs an NVIDIA GPU to run in real time.
)

echo [3/5] Face service environment (large download, takes a while)...
if not exist face-service\venv ( python -m venv face-service\venv || goto :fail )
call face-service\venv\Scripts\python.exe -m pip install --upgrade pip || goto :fail
call face-service\venv\Scripts\python.exe -m pip install -r face-service\requirements.txt || goto :fail

echo [4/5] Frontend packages...
pushd frontend
call npm install || ( popd & goto :fail )
popd

echo [5/5] Installing bundled AI model caches (skips files that already exist)...
robocopy "_model_cache" "%USERPROFILE%" /E /XC /XN /XO /NFL /NDL /NJH /NJS /NP >nul
if errorlevel 8 goto :fail

echo.
echo Setup complete. Run start_all.bat to launch everything.
exit /b 0

:fail
echo.
echo Setup FAILED - see the messages above.
exit /b 1
