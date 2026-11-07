#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
for i in $(seq 1 95); do
  if grep -q 'ROLES_DONE' sim_logs/roles.log 2>/dev/null; then break; fi
  sleep 6
done
echo "roles seeds: $(wc -l < v2_roles.jsonl 2>/dev/null)"
if grep -q 'ROLES_DONE' sim_logs/roles.log 2>/dev/null; then
  tail -n 30 sim_logs/roles.log
else
  echo "(still running)"
fi
