"""Make NNInterface::getObservation robust to an unset/unmapped boss.

Root cause: `bossEncodeMap.at(gc.boss)` throws std::out_of_range (surfaced to
Python as IndexError 'map::at') early in a run when gc.boss is INVALID/not yet
assigned. Fix mirrors the existing getCardIdx fallback: look up safely and skip
the boss one-hot when the boss isn't mapped.
"""
import shutil
import sys

p = "bindings/bindings-util.cpp"
src = open(p).read()

old = (
    "        int bossEncodeIdx = offset + bossEncodeMap.at(gc.boss);\n"
    "        ret[bossEncodeIdx] = 1;\n"
)
new = (
    "        // robustness: gc.boss can be INVALID/unset early in a run; skip the\n"
    "        // boss one-hot instead of throwing (mirrors getCardIdx fallback).\n"
    "        auto bossIt = bossEncodeMap.find(gc.boss);\n"
    "        if (bossIt != bossEncodeMap.end()) {\n"
    "            ret[offset + bossIt->second] = 1;\n"
    "        }\n"
)

if new in src:
    print("ALREADY_PATCHED")
    sys.exit(0)
if old not in src:
    print("ANCHOR_NOT_FOUND -- aborting, no change made")
    sys.exit(1)

shutil.copyfile(p, p + ".orig")
open(p, "w").write(src.replace(old, new, 1))
print("PATCHED (backup at %s.orig)" % p)
