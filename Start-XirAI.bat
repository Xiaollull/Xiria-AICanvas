@echo off
setlocal
cd /d "%~dp0"
where node >nul 2>nul
if errorlevel 1 (
  echo Startup failed: Node.js is missing.
  echo Please run Setup-XirAI first.
  pause
  exit /b 1
)
echo Cleaning existing XiriaCanvas AI processes...
node "scripts\cleanup-processes.mjs"
if errorlevel 1 (
  echo Startup stopped because existing processes could not be cleaned.
  pause
  exit /b 1
)
node "scripts\start.mjs"
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" echo XiriaCanvas AI exited with code %EXIT_CODE%.
echo Press any key to close this window.
pause >nul
exit /b %EXIT_CODE%
