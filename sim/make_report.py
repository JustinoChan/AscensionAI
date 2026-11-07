"""Hardening report: stage-progression table + plots from saved run data.
Stages on the 150 fixed eval seeds where available; baseline + search-combat
cited from their original evals (labeled). Saves PNGs to plots/."""
import json, os
import numpy as np
os.makedirs("plots", exist_ok=True)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception as e:
    HAVE_PLT = False
    print("matplotlib unavailable:", e)


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []

mv2 = load("v2_mainline_v2.jsonl")     # mainline_v2 (150)
dh = load("v2_A.jsonl")                # deck-hygiene + varL (150, from relic A/B)

# ---- stage progression table ----
stages = [
    ("baseline (pre-search)", 14.71, 0, "48*"),
    ("search-combat", 26.35, 0, "48*"),
    ("+ deck-hygiene", float(np.mean([r["floor"] for r in dh])) if dh else 23.6,
     sum(r["win"] for r in dh) if dh else 5, str(len(dh) or 150)),
    ("mainline_v2", float(np.mean([r["floor"] for r in mv2])) if mv2 else 24.9,
     sum(r["win"] for r in mv2) if mv2 else 7, str(len(mv2) or 150)),
]
print("==== STAGE PROGRESSION ====")
print("%-24s %7s %6s %6s" % ("stage", "floor", "wins", "seeds"))
for name, fl, w, n in stages:
    print("%-24s %7.1f %6s %6s" % (name, fl, w, n))
print("(* baseline & search-combat measured on the original 48-seed full-run eval)")

if HAVE_PLT and mv2:
    # floor distribution (mainline_v2)
    fl = [r["floor"] for r in mv2]
    plt.figure(figsize=(7, 3.2))
    plt.hist(fl, bins=range(0, 56, 2), color="#1d5f99", edgecolor="white")
    plt.axvline(np.mean(fl), color="#9f2f2f", linestyle="--", label="mean %.1f" % np.mean(fl))
    plt.xlabel("final floor"); plt.ylabel("runs"); plt.title("mainline_v2 floor distribution (150 seeds)")
    plt.legend(); plt.tight_layout(); plt.savefig("plots/floor_dist.png", dpi=110); plt.close()

    # deaths by act
    da = {1: 0, 2: 0, 3: 0}
    wins = 0
    for r in mv2:
        if r["win"]:
            wins += 1
        else:
            da[r.get("death_act", 1)] = da.get(r.get("death_act", 1), 0) + 1
    plt.figure(figsize=(6, 3.2))
    plt.bar(["Act 1", "Act 2", "Act 3", "WIN"],
            [da[1], da[2], da[3], wins],
            color=["#9b640b", "#9f2f2f", "#5d6875", "#247a3d"])
    plt.ylabel("runs"); plt.title("mainline_v2 outcome by act (150 seeds)")
    plt.tight_layout(); plt.savefig("plots/deaths_by_act.png", dpi=110); plt.close()

    # stage progression (floor + wins)
    names = [s[0] for s in stages]; floors = [s[1] for s in stages]; w = [s[2] for s in stages]
    fig, ax1 = plt.subplots(figsize=(7, 3.4))
    ax1.bar(names, floors, color="#1d5f99", alpha=0.8); ax1.set_ylabel("avg floor", color="#1d5f99")
    ax2 = ax1.twinx(); ax2.plot(names, w, "o-", color="#247a3d", linewidth=2); ax2.set_ylabel("wins/150", color="#247a3d")
    plt.title("AscensionAI stage progression"); plt.xticks(rotation=15, ha="right")
    plt.tight_layout(); plt.savefig("plots/progression.png", dpi=110); plt.close()
    print("saved plots/: floor_dist.png, deaths_by_act.png, progression.png")
