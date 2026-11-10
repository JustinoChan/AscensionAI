"""Auto-generate docs/experiments/018_rl_round3.md from the round-3 history and gate files (refined by hand later)."""
import json, os, time
R = "/mnt/c/Code/AscensionAI/docs/experiments/018_rl_round3.md"
h = json.load(open("nets/rl3/train_history.json"))
bc = open("nets/rl3/bc.log").read().strip().splitlines()[-1] if os.path.exists("nets/rl3/bc.log") else ""
def read(p):
    return open(p).read().strip() if os.path.exists(p) else "(not available yet)"
L = ["# 018 RL Round 3: Visible-Map Features + Boss-Kill Bonus", "",
     "**Status:** auto-generated %s from the run logs; numbers are final, commentary to follow." % time.strftime("%F %H:%M %Z"), "",
     "## Setup", "",
     "Same candidate-scorer PPO as 017, with two changes:",
     "",
     "- **`MACRO_MAPFEAT=1`:** 28 extra inputs summarising the visible current-act map.",
     "  - For each room type, the min and max count over the paths to the boss, plus floors left and the number of paths.",
     "  - Given for the position now and after each option.",
     "- **`RL_BOSS_BONUS=0.3`:** reward = floor/50 + win + 0.3 per act boss beaten.",
     "",
     "**Data and training:**",
     "",
     "- Fresh behaviour-cloning data: 1000 K=8 runs, seeds 820000013+19i.",
     "- PPO: 150 iterations × 100 games at K=8 on 14 workers, seeds 640000007+13i.",
     "- Same 100 validation seeds as 017.", "",
     "BC: `%s`" % bc[:200], "", "## Validation (greedy, K=8, 100 seeds)", "", "| checkpoint | floor | wins |", "|---|---|---|"]
L += ["| %s | %.2f | %d |" % (v["tag"], v["floor"], v["wins"]) for v in h["rl_val"]]
L += ["", "017 reference on the same seeds: it099 = 29.65 floor, 7 wins; heuristics = 21.39, 0.", "",
      "## eval500 gate (honest K=64)", "", "```", read("sim_logs/GATE_rl3_eval500.txt"), "```", "",
      "## Overnight branch", "", "```", read("sim_logs/NIGHT_BRANCH.txt"), "```"]
for f in ("sim_logs/GATE_holdout_rl3_K64.txt", "sim_logs/GATE_rl3c_eval500.txt", "sim_logs/GATE_rl4_eval500.txt"):
    if os.path.exists(f):
        L += ["", "### %s" % os.path.basename(f), "", "```", read(f), "```"]
open(R, "w").write("\n".join(L) + "\n")
print("wrote", R)
