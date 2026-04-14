@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found at .venv\Scripts\python.exe
  exit /b 1
)

echo Installing packaging dependencies...
"%~dp0.venv\Scripts\python.exe" -m pip install -r requirements.txt pyinstaller
if errorlevel 1 exit /b 1

echo Cleaning previous build artifacts...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo Building desktop executable...
"%~dp0.venv\Scripts\pyinstaller.exe" --clean TimeAndAttendance.spec
if errorlevel 1 exit /b 1

echo Build complete.
echo Output folder: %~dp0dist\TimeAndAttendance
