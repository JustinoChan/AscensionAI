"""Expose BattleContext::init(const GameContext&, MonsterEncounter) -- the
overload already exists in C++ (include/combat/BattleContext.h:102); only the
binding is missing (bindings/slaythespire.cpp:144 binds the one-arg form)."""
import shutil

P = "bindings/slaythespire.cpp"
ONE = '.def("init", pybind11::overload_cast<const GameContext &>(&BattleContext::init))'
TWO = '.def("init", pybind11::overload_cast<const GameContext &, MonsterEncounter>(&BattleContext::init))'

src = open(P).read()
if TWO in src:
    print("already patched")
    raise SystemExit(0)
assert ONE in src, "anchor line not found -- bindings changed?"
shutil.copy(P, P + ".orig3")
src = src.replace(ONE, ONE + "\n        " + TWO, 1)
open(P, "w").write(src)
print("patched (backup: %s.orig3)" % P)
