@echo off
setlocal
cd /d "%~dp0"
where node >nul 2>nul
if errorlevel 1 (
  echo Node.js 22.21 or newer is required.
  echo Please install Node.js and double-click this file again.
  pause
  exit /b 1
)
echo XiriaCanvas AI Environment Setup
echo A browser window will open. Choose Auto or Manual setup, then click Start.
echo The wizard queries the PyTorch catalog first - nothing is installed yet.
node "scripts\setup-gui.mjs"
if errorlevel 1 pause