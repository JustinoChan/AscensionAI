' AscensionAI: keep WSL alive and resume the sim job queue after a reboot + logon.
' Install (once):  copy this file into your Startup folder (Win+R -> shell:startup).
' Remove it from there to disable. Runs hidden; it starts ~/sts/keepalive.sh in WSL,
' which resumes the queue immediately and re-checks it every 15 minutes.
Set sh = CreateObject("WScript.Shell")
sh.Run "wsl.exe -d Ubuntu -- bash -lc ""~/sts/keepalive.sh""", 0, False
