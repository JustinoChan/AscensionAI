#!/bin/bash
# Long-running WSL keep-alive + watchdog (started hidden at logon by tools/AscensionAI_keepalive.vbs).
# Resumes the queue right away, then re-checks every 15 min. One instance only.
exec 9> /tmp/ascension_keepalive.lock
flock -n 9 || exit 0
while true; do
  ~/sts/overnight_watchdog.sh
  sleep 900
done
