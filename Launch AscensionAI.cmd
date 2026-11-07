@echo off
rem Opens the AscensionAI control panel (Live Game + Sim Experiments tabs).
rem .pyw files have no file association on this laptop, so launch pythonw directly.
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" "%~dp0AscensionAI.pyw"
) else if exist "%LOCALAPPDATA%\Programs\Python\Python312\pythonw.exe" (
  start "" "%LOCALAPPDATA%\Programs\Python\Python312\pythonw.exe" "%~dp0AscensionAI.pyw"
) else (
  start "" pyw -3 "%~dp0AscensionAI.pyw"
)
