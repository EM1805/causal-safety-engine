@echo off
setlocal
cd /d %~dp0

REM PCB demo (Windows)
python demo.py

echo.
echo Done. Outputs are in the "out" folder.
pause
