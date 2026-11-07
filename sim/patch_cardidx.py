"""Make NNInterface::getCardIdx memory-safe (fixes segfault under deep states).

Two unchecked operations cause out-of-bounds memory access:
  1. `cardEncodeMap[static_cast<int>(id)]` -- OOB *read* if id >= map size.
  2. returned idx feeds `ret[offset+idx]` writes into 220-wide card regions with
     no upper-bound check, so idx>=220 corrupts adjacent memory -> segfault.
Add a read guard and a region-bounds guard (returns 0 fallback, like the
existing rawIdx==-1 path).
"""
import shutil
import sys

p = "bindings/bindings-util.cpp"
src = open(p).read()

old_read = "        int rawIdx = cardEncodeMap[static_cast<int>(id)];\n"
new_read = (
    "        int _ci = static_cast<int>(id);\n"
    "        if (_ci < 0 || _ci >= static_cast<int>(cardEncodeMap.size())) return 0; // OOB-read guard\n"
    "        int rawIdx = cardEncodeMap[_ci];\n"
)

old_ret = (
    "        int idx = rawIdx * 2;\n"
    "        if (upgradeCount > 0) {\n"
    "            idx += 1;\n"
    "        }\n"
    "\n"
    "        return idx;\n"
)
new_ret = (
    "        int idx = rawIdx * 2;\n"
    "        if (upgradeCount > 0) {\n"
    "            idx += 1;\n"
    "        }\n"
    "        if (idx < 0 || idx >= 220) return 0; // region-bounds guard (each card region is 220 wide)\n"
    "        return idx;\n"
)

if new_read in src and new_ret in src:
    print("ALREADY_PATCHED"); sys.exit(0)
if old_read not in src or old_ret not in src:
    print("ANCHOR_NOT_FOUND read=%s ret=%s" % (old_read in src, old_ret in src)); sys.exit(1)

shutil.copyfile(p, p + ".orig2")
src = src.replace(old_read, new_read, 1).replace(old_ret, new_ret, 1)
open(p, "w").write(src)
print("PATCHED getCardIdx (backup .orig2)")
