#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
cp bank_eval.jsonl bank_eval.jsonl.16boss.bak
rm -f harvest_bosses.jsonl
# ~160 fresh varE greedy runs; seeds 2,000,000+ (idx offset 1,000,000), disjoint
for s in $(seq 0 7); do
  lo=$((1000000 + s*20)); hi=$((lo + 20))
  for t in 1 2 3 4 5; do
    python harvest_bank.py policy_varE.npz $lo $hi harvest_bosses.jsonl && break
    echo "retry seg $t (block $lo)"
  done
done
echo "harvested total: $(wc -l < harvest_bosses.jsonl)  bosses: $(grep -c '\"room\": \"BOSS\"' harvest_bosses.jsonl)"
echo "=== merge ==="
python merge_eval_bosses.py
echo "=== export varJ policy for baseline ==="
python export_ckpt.py sim_models_varJ/ppo_sim_latest.pt policy_varJ.npz
echo "=== baseline greedy on expanded bank (varM start = varJ; varL for reference) ==="
for t in 1 2 3 4 5; do python eval_greedy_bank.py policy_varJ.npz && break; echo retry $t; done
for t in 1 2 3 4 5; do python eval_greedy_bank.py policy_varL.npz && break; echo retry $t; done
