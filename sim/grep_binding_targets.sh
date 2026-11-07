#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
echo "===== DamageInfo struct (full) ====="
sed -n '20,48p' include/combat/Monster.h
echo "===== Monster pybind block in slaythespire.cpp ====="
grep -n "py::class_<Monster\|pybind11::class_<Monster\|class_<sts::Monster\|Monster>" bindings/slaythespire.cpp
awk '/class_<Monster|class_<sts::Monster/{f=1} f{print NR": "$0} /;/{if(f && /\);/){f=0}}' bindings/slaythespire.cpp | head -30
echo "===== how Monster is currently bound (context) ====="
grep -n "Monster\|monsters\|is_alive\|cur_hp" bindings/slaythespire.cpp | head -30
echo "===== MonsterStatus enum (vulnerable/weak/strength indices) ====="
MS=$(grep -rl "enum class MonsterStatus" include/ src/ | head -1)
echo "file: $MS"
awk '/enum class MonsterStatus/,/};/' "$MS"
echo "===== PlayerStatus enum ====="
PS=$(grep -rl "enum class PlayerStatus" include/ src/ | head -1)
echo "file: $PS"
awk '/enum class PlayerStatus/,/};/' "$PS"
echo "===== getStatusInternal / getStatusRuntime signatures ====="
grep -rn "getStatusInternal\|getStatusRuntime" include/combat/*.h
echo "===== Player pybind block ====="
grep -n "class_<Player\|class_<sts::Player\|player\b" bindings/slaythespire.cpp | head -20
