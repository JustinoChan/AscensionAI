"""
make_eval_seeds.py - Generate deterministic seed files for controlled evals.

Usage:
  python scripts/make_eval_seeds.py --count 200 --out seeds/eval_200.txt
"""

from __future__ import annotations

import argparse
import os
import random


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--out", type=str, default="seeds/eval_200.txt")
    parser.add_argument("--rng-seed", type=int, default=20260506)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    rng = random.Random(args.rng_seed)
    seen: set[str] = set()
    seeds: list[str] = []
    while len(seeds) < args.count:
        seed = str(rng.randrange(1, 2**31 - 1))
        if seed not in seen:
            seen.add(seed)
            seeds.append(seed)

    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# AscensionAI fixed eval seeds; rng_seed={args.rng_seed}\n")
        for seed in seeds:
            f.write(seed + "\n")
    print(out)


if __name__ == "__main__":
    main()
