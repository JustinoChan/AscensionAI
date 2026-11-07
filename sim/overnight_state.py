"""State + dashboard sync for overnight.sh.
  overnight_state.py init | get <job> | set <job> <state> [note] | gate <train_report.json>
"""
import json, os, sys, time

D = "sim_logs/overnight"
P = D + "/state.json"
JOBS = [("harvest", "a. harvest mainline_h1 K=64, 1600 training seeds", None),
        ("train", "b. train value net + ranking prior (GPU) + offline gate", None),
        ("c1_prior3_K64", "c1. A/B: prior top-3, K=64", "ab_prior3_K64"),
        ("c2_pv_d8_K64", "c2. A/B: prior top-3 + V leaf depth 8, K=64", "ab_pv_d8_K64"),
        ("c3_pv_d8_Keq", "c3. A/B: prior top-3 + V leaf depth 8, K at equal wall-clock", "ab_pv_d8_Keq"),
        ("c4_pv_d16_K64", "c4. A/B: prior top-3 + V leaf depth 16, K=64", "ab_pv_d16_K64"),
        ("d_holdout_K128", "d. holdout-150 K=128 check", "holdout_K128"),
        ("report", "e. OVERNIGHT_REPORT.md + dashboard export + commit/push", None)]


def load():
    return json.load(open(P)) if os.path.exists(P) else None


def save(s):
    os.makedirs(D, exist_ok=True)
    json.dump(s, open(P + ".t", "w"), indent=1); os.replace(P + ".t", P)
    sync(s)


def sync(s):
    # plan checklist (Sim Experiments tab)
    pp = "sim_logs/PHASES.json"
    ph = json.load(open(pp))
    ph = [x for x in ph if x.get("name") != "Overnight queue (Nov 7)"]
    allst = [j["state"] for j in s["jobs"]]
    ph.append({"name": "Overnight queue (Nov 7)",
               "state": "done" if all(x in ("done", "failed", "skipped") for x in allst) else "running",
               "steps": [{"name": j["name"] + (" — " + j["note"] if j.get("note") else ""),
                          "state": {"todo": "todo", "running": "running", "done": "done"}.get(j["state"], "done" if j["state"] == "skipped" else "todo")}
                         for j in s["jobs"]]})
    json.dump(ph, open(pp + ".t", "w"), indent=1); os.replace(pp + ".t", pp)
    # queue progress bars (seed-level)
    runs = [j["arm"] for j in s["jobs"] if j.get("arm")]
    q = {"name": "Overnight queue (Nov 7)", "runs": ["harvest_h1"] + runs,
         "totals": dict([("harvest_h1", 1600)] + [(r, 150) for r in runs]),
         "est_secs": {"harvest_h1": 18000, "ab_prior3_K64": 1800, "ab_pv_d8_K64": 900, "ab_pv_d8_Keq": 3400,
                      "ab_pv_d16_K64": 1200, "holdout_K128": 6000}}
    json.dump(q, open("sim_logs/queues/overnight.json", "w"), indent=1)


cmd = sys.argv[1]
s = load()
if cmd == "init":
    if s is None:
        s = {"created": time.time(), "jobs": [{"id": i, "name": n, "arm": a, "state": "todo", "note": ""} for i, n, a in JOBS]}
    for j in s["jobs"]:
        if j["state"] == "running":      # runner died mid-job: resume it
            j["state"] = "todo"
    save(s)
elif cmd == "get":
    print(next(j["state"] for j in s["jobs"] if j["id"] == sys.argv[2]))
elif cmd == "set":
    for j in s["jobs"]:
        if j["id"] == sys.argv[2]:
            if sys.argv[3] == "done" and j["state"] == "skipped":
                continue                  # a job that skipped itself stays skipped
            j["state"] = sys.argv[3]
            if len(sys.argv) > 4 and sys.argv[4]:
                j["note"] = sys.argv[4]
            j["t_" + sys.argv[3]] = time.time()
            if sys.argv[3] in ("done", "failed", "skipped") or (sys.argv[3] == "running"):
                os.makedirs("sim_logs/jobs", exist_ok=True)
                with open("sim_logs/jobs/events.jsonl", "a") as f:
                    f.write(json.dumps({"t": time.time(), "id": "overnight:" + j["id"], "name": j["name"], "state": j["state"]}) + chr(10))
    save(s)
elif cmd == "gate":
    r = json.load(open(sys.argv[2]))
    pv, bv, vv = r["prior_val"], r["varN_val"], r["value_val"]
    # prior: must rank the K=64 Q better than varN at the pruning width (top-3)
    prior = pv["regret@3"] < bv["regret@3"] and pv["recall@3"] >= bv["recall@3"]
    # value: clearly better than a constant leaf (MAE <= 0.8x) and positive R^2
    value = vv["V_mae"] <= 0.8 * vv["V_mae_const"] and vv["V_r2"] > 0.2
    g = {"prior": prior, "value": value,
         "why": "prior: regret@3 %.3f vs varN %.3f, recall@3 %.3f vs varN %.3f (need both better); value: V_mae %.2f vs const %.2f (need <=0.8x), V_r2 %.3f (need >0.2), win ECE %.3f" % (
             pv["regret@3"], bv["regret@3"], pv["recall@3"], bv["recall@3"], vv["V_mae"], vv["V_mae_const"], vv["V_r2"], vv["win_ece"])}
    json.dump(g, open(os.path.join(os.path.dirname(sys.argv[2]), "GATE.json"), "w"), indent=1)
    for j in s["jobs"]:
        if j["id"] == "train":
            j["note"] = ("prior PASS" if prior else "prior FAIL") + ", " + ("value PASS" if value else "value FAIL")
    save(s)
    print(g)
