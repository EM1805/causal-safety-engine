@echo off
setlocal
cd /d %~dp0

REM PCB demo (Windows)
python demo.py
python ..\..\tools\authority_build.py --out out
python ..\..\tools\do_readiness.py --out out
python ..\..\tools\authority_propagate_do_check.py --out out
python ..\..\tools\authority_propagate_do_check.py --out out

echo.
echo Done. Outputs are in the "out" folder.
pause
