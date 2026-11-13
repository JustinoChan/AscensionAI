#!/bin/bash
# Export ledger + digest + gate to the website repo and push the branch (Windows git via interop).
cd "$HOME/sts"
R=/mnt/c/Code/AscensionAI
nice -n 10 python3 ledger.py ingest > /dev/null
python3 ledger.py export $R/docs/experiments
.venv/bin/python dash_data.py digest --write > /dev/null
D=$(ls -t sim_logs/DIGEST_*.md | head -1); cp "$D" "$R/docs/experiments/digest_$(basename $D .md | cut -d_ -f2).md"
for g in sim_logs/GATE_*.txt; do cp "$g" $R/sim/; done; for d in rl2 rl3 rl3c rl3d rl3e rl4a rl4b rl5 rl5b rl5c; do [ -f nets/$d/train_history.json ] && mkdir -p $R/sim/nets_$d && cp nets/$d/train_history.json $R/sim/nets_$d/ && { cp nets/$d/pol_gate.npz $R/sim/nets_$d/ 2>/dev/null; true; }; done; cp nets/rollout1/report.json $R/sim/rollout1_report.json 2>/dev/null
for f in honest.py stack_and_gate.sh push_results.sh jobq.py dash_data.py rl_macro_train.py rl_gate.sh holdout_rl.sh holdout_pol.sh ppo2.sh ppo3d.sh rl4_ablation.sh rl3_pipeline.sh night_branch.sh make018.py mapfeat.py boss_ab.sh boss_report.py train_rollout.py run_status.py; do cp $f $R/sim/$f; done
G="/mnt/c/Program Files/Git/cmd/git.exe"
BR=""; for try in 1 2 3 4 5 6; do BR=$("$G" -C 'C:\Code\AscensionAI' rev-parse --abbrev-ref HEAD 2>/dev/null); [ -n "$BR" ] && break; sleep 20; done   # interop can fail transiently
[ "$BR" = "claude/project-thread-p8o8l7" ] || { echo "website repo on $BR - not committing"; exit 1; }
for try in 1 2 3 4 5; do "$G" -C 'C:\Code\AscensionAI' status --short >/dev/null 2>&1 && break; sleep 20; done   # WSL interop can fail transiently
"$G" -C 'C:\Code\AscensionAI' add docs/experiments sim
"$G" -C 'C:\Code\AscensionAI' commit -q -m "${MSG:-Sim results (auto-generated)}

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || true
for try in 1 2 3 4 5; do "$G" -C 'C:\Code\AscensionAI' push -q origin claude/project-thread-p8o8l7 && break; sleep 30; done
