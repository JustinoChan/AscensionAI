#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
echo "== MonsterMoveId enum size =="
F=$(grep -rl "enum class MonsterMoveId" include/ src/ 2>/dev/null | head -1)
echo "file: $F"
awk '/enum class MonsterMoveId/,/};/' "$F" | wc -l
awk '/enum class MonsterMoveId/,/};/' "$F" | tail -8
echo "== where moveHistory[0] is set (rollMove timing) =="
grep -rn "moveHistory\[0\]" include/ src/ | head -10
echo "== GREMLIN_NOB move ids =="
awk '/enum class MonsterMoveId/,/};/' "$F" | grep -n "GREMLIN_NOB"
echo "== run live intent probe =="
source .venv/bin/activate
python probe_intent.py
