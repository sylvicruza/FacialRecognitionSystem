@echo off
setlocal
cd /d "%~dp0"

for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":8000 .*LISTENING"') do (
  taskkill /PID %%p /F >nul 2>&1
)

set DJANGO_SETTINGS_MODULE=ohc_time_attendance.settings
set ATTENDANCE_DESKTOP_MODE=1
"%~dp0.venv\Scripts\python.exe" run_attendance_client.py
