#!/bin/bash
# Export ledger + digest + gate to the website repo and push the branch (Windows git via interop).
cd "$HOME/sts"
R=/mnt/c/Code/AscensionAI
nice -n 10 python3 ledger.py ingest > /dev/null
python3 ledger.py export $R/docs/experiments
.venv/bin/python dash_data.py digest --write > /dev/null
D=$(ls -t sim_logs/DIGEST_*.md | head -1); cp "$D" "$R/docs/experiments/digest_$(basename $D .md | cut -d_ -f2).md"
[ -f sim_logs/GATE_phase3_2.txt ] && cp sim_logs/GATE_phase3_2.txt $R/sim/GATE_phase3_2.txt
for f in honest.py stack_and_gate.sh push_results.sh jobq.py dash_data.py; do cp $f $R/sim/$f; done
G="/mnt/c/Program Files/Git/cmd/git.exe"
BR=$("$G" -C 'C:\Code\AscensionAI' rev-parse --abbrev-ref HEAD)
[ "$BR" = "claude/project-thread-p8o8l7" ] || { echo "website repo on $BR - not committing"; exit 1; }
"$G" -C 'C:\Code\AscensionAI' add docs/experiments sim
"$G" -C 'C:\Code\AscensionAI' commit -q -m "Phase 3 step 2c results (card-search fix, stack with rest search, holdout gate) - auto-generated

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || true
"$G" -C 'C:\Code\AscensionAI' push -q origin claude/project-thread-p8o8l7
