#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
echo "== encounter in bindings =="
grep -rn "encounter" bindings/ | head -25
echo "== init bindings =="
grep -rn '"init"' bindings/ | head -10
echo "== BattleContext init overloads in C++ =="
find . -name "BattleContext.h" -not -path "./.venv/*" | head -3
find . -name "BattleContext.h" -not -path "./.venv/*" -exec grep -n "void init" {} \;
echo "== GameContext encounter member =="
find . -name "GameContext.h" -not -path "./.venv/*" -exec grep -n "MonsterEncounter" {} \; | head -10
