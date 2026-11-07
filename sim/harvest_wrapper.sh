#!/bin/bash
# Restart-on-crash wrapper: some seeds deterministically segfault the fork
# under greedy play (known from eval). Skip the crashing seed and continue.
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
START=${1:-0}
END=${2:-300}
while true; do
  python harvest_bank.py policy_varE.npz "$START" "$END" bank_raw.jsonl
  CODE=$?
  if [ "$CODE" -eq 0 ]; then break; fi
  LAST=$(cat .harvest_progress 2>/dev/null || echo "$START")
  START=$((LAST + 1))
  echo "harvest crashed (exit $CODE) at idx $LAST; resuming from $START"
  if [ "$START" -ge "$END" ]; then break; fi
done
echo "wrapper done"
