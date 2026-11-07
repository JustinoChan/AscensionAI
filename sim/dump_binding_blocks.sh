#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
echo "===== bindings/slaythespire.cpp lines 110-160 ====="
sed -n '110,160p' bindings/slaythespire.cpp
echo "===== includes at top ====="
sed -n '1,30p' bindings/slaythespire.cpp | grep -n "#include"
echo "===== is MonsterStatusEffects.h included anywhere reachable? ====="
grep -rn "MonsterStatusEffects\|MonsterStatus" bindings/slaythespire.cpp include/combat/Monster.h | head
