#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
echo "==== npz policies present ===="
ls -la --time-style=+%H:%M:%S *.npz 2>/dev/null
echo
echo "==== varE / varL checkpoints ===="
ls -la sim_models_varE/ppo_sim_latest.pt 2>/dev/null
ls -la sim_models_varL/ 2>/dev/null | tail -4
echo
echo "==== full-run env: where is the StsEnv / full-run worker? ===="
grep -rln "class .*Env" --include=*.py . | head -20
echo
echo "---- StsEnv-ish class defs ----"
grep -rn "class .*Env" --include=*.py . | grep -vi fight | head
echo
echo "==== does the full-run worker track action history / expose in_combat? ===="
grep -rn "in_combat\|self.live\|action_history\|def reset\|def step\|screen_state\|ScreenState.BATTLE" sim_worker.py 2>/dev/null | head -30
echo
echo "==== varE NumPy forward (non-combat policy) — export + forward code ===="
ls -la export_policy_npz.py harvest_bank.py 2>/dev/null
grep -n "npz\|def .*forward\|gelu\|tanh\|pbody\|\.npy\|OBS\|1200\|1235\|matmul\|@ \|np.dot" harvest_bank.py 2>/dev/null | head -40
echo
echo "==== combat_block out-of-combat behavior ===="
grep -n "def combat_block\|bc is None\|in_combat\|return\|zeros\|K =" combat_features.py 2>/dev/null | head -30
