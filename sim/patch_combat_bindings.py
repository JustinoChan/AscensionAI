"""Add read-only combat queries to the Monster and Player pybind blocks.
Idempotent; backs up to .orig4 on first run."""
import shutil

P = "bindings/slaythespire.cpp"
src = open(P).read()

MARKER = '.def("intent_base_damage"'
if MARKER in src:
    print("already patched")
    raise SystemExit(0)

PLAYER_ANCHOR = '        .def_readwrite("block", &Player::block);'
PLAYER_NEW = (
    '        .def_readwrite("block", &Player::block)\n'
    '        .def("vulnerable", [](const Player &p){ '
    'return p.getStatusRuntime(PlayerStatus::VULNERABLE); })\n'
    '        .def("weak", [](const Player &p){ '
    'return p.getStatusRuntime(PlayerStatus::WEAK); })\n'
    '        .def("strength", [](const Player &p){ '
    'return p.getStatusRuntime(PlayerStatus::STRENGTH); });'
)

MONSTER_ANCHOR = '        .def("is_alive", &Monster::isAlive);'
MONSTER_NEW = (
    '        .def("is_alive", &Monster::isAlive)\n'
    '        .def("is_attacking", &Monster::isAttacking)\n'
    '        .def("intent_base_damage", [](const Monster &mon, '
    'const BattleContext &bc) {\n'
    '            if (!mon.isAttacking()) return std::make_pair(0, 0);\n'
    '            auto di = mon.getMoveBaseDamage(bc);\n'
    '            return std::make_pair(di.damage, di.attackCount);\n'
    '        })\n'
    '        .def("vulnerable", [](const Monster &mon){ '
    'return mon.getStatusInternal(MonsterStatus::VULNERABLE); })\n'
    '        .def("weak", [](const Monster &mon){ '
    'return mon.getStatusInternal(MonsterStatus::WEAK); })\n'
    '        .def("strength", [](const Monster &mon){ '
    'return mon.getStatusInternal(MonsterStatus::STRENGTH); });'
)

assert PLAYER_ANCHOR in src, "player anchor not found"
assert MONSTER_ANCHOR in src, "monster anchor not found"
shutil.copy(P, P + ".orig4")
src = src.replace(PLAYER_ANCHOR, PLAYER_NEW, 1)
src = src.replace(MONSTER_ANCHOR, MONSTER_NEW, 1)
open(P, "w").write(src)
print("patched (backup %s.orig4)" % P)
