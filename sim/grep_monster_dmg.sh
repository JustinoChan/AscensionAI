#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
MF=$(find . -name "Monster.h" -path "*include*" | head -1)
echo "Monster.h: $MF"
echo "===== damage / intent / move methods on Monster ====="
grep -n "Damage\|damage\|getIntent\|intentDmg\|moveHistory\|isAttack\|getMove\|calculateMove\|baseDamage\|moveBaseDamage\|monsterTurn" "$MF" | head -40
echo "===== isMoveAttack / move-damage helpers anywhere ====="
grep -rn "getIntentBaseDamage\|getMoveBaseDamage\|isMoveAttack\|calculateMoveDamage\|intentDamage\|getMoveDamage" include/ src/ | head -25
echo "===== MonsterMoveDamage.cpp head (how damage per move is computed) ====="
sed -n '1,60p' src/combat/MonsterMoveDamage.cpp
echo "===== does Monster expose attack count / adjustedDamage? ====="
grep -n "adjustedDamage\|moveDamage\|numAttacks\|multiAttack\|timesAttack\|getAttackCount" "$MF"
