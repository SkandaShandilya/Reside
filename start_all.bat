@echo off
REM Starts the face service (8010), the main backend (8000) and the frontend (5173).
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo Run setup.bat first.
  exit /b 1
)

start "RESIDE face service :8010" /D "%~dp0face-service" cmd /k "venv\Scripts\python.exe -m uvicorn main:app --port 8010"
start "RESIDE backend :8000" /D "%~dp0backend" cmd /k "..\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000"
start "RESIDE frontend :5173" /D "%~dp0frontend" cmd /k "npm run dev"

echo Started. The backend takes a minute to load the AI models.
echo Open http://localhost:5173
