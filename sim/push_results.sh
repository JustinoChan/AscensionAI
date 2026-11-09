#!/bin/bash
# Export ledger + digest + gate to the website repo and push the branch (Windows git via interop).
cd "$HOME/sts"
R=/mnt/c/Code/AscensionAI
nice -n 10 python3 ledger.py ingest > /dev/null
python3 ledger.py export $R/docs/experiments
.venv/bin/python dash_data.py digest --write > /dev/null
D=$(ls -t sim_logs/DIGEST_*.md | head -1); cp "$D" "$R/docs/experiments/digest_$(basename $D .md | cut -d_ -f2).md"
for g in sim_logs/GATE_*.txt; do cp "$g" $R/sim/; done; mkdir -p $R/sim/nets_rl2; cp nets/rl2/train_history.json nets/rl2/pol_gate.npz $R/sim/nets_rl2/ 2>/dev/null; cp nets/rollout1/report.json $R/sim/rollout1_report.json 2>/dev/null
for f in honest.py stack_and_gate.sh push_results.sh jobq.py dash_data.py rl_macro_train.py rl_gate.sh holdout_rl.sh ppo2.sh train_rollout.py run_status.py; do cp $f $R/sim/$f; done
G="/mnt/c/Program Files/Git/cmd/git.exe"
BR=$("$G" -C 'C:\Code\AscensionAI' rev-parse --abbrev-ref HEAD)
[ "$BR" = "claude/project-thread-p8o8l7" ] || { echo "website repo on $BR - not committing"; exit 1; }
"$G" -C 'C:\Code\AscensionAI' add docs/experiments sim
"$G" -C 'C:\Code\AscensionAI' commit -q -m "${MSG:-Sim results (auto-generated)}

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || true
"$G" -C 'C:\Code\AscensionAI' push -q origin claude/project-thread-p8o8l7
