@echo off
REM ==========================================================
REM  Sport AI Lab Start - avvio su Windows
REM  Doppio clic su questo file. Non serve VS Code.
REM  Fa tutto avvia.py: questo file serve solo al doppio clic.
REM ==========================================================
cd /d "%~dp0"
chcp 65001 >nul

where py >nul 2>&1
if errorlevel 1 (
  python avvia.py %*
) else (
  py avvia.py %*
)

if errorlevel 1 (
  echo.
  echo [ERRORE] Python non risulta installato o non e' nel PATH.
  echo Installalo da https://www.python.org/downloads/
  echo ricordando di spuntare "Add Python to PATH".
  pause
)
