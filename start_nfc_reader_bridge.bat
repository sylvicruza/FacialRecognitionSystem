@echo off
setlocal
cd /d "%~dp0"
set DJANGO_SETTINGS_MODULE=ohc_time_attendance.settings
set ATTENDANCE_DESKTOP_MODE=1
"%~dp0.venv\Scripts\python.exe" run_acr122u_bridge.py
