"""General sim job queue (runs after / alongside nothing: one job at a time).

  python3 jobq.py list [--json]
  python3 jobq.py add <name> -- <shell command...>          append a raw job
  python3 jobq.py add-preset <preset> <arm> "<ENV=..>" [desc] append from a preset
  python3 jobq.py presets
  python3 jobq.py cancel <id> | move <id> up|down | retry <id>
  python3 jobq.py run          runner (started by the watchdog; one instance; waits for
                               the overnight runner to finish before starting anything)
State: sim_logs/jobs/queue.json   Log: sim_logs/jobs/jobq.log
Jobs that run 150-seed arms are shown with progress bars in the dashboard
(sim_logs/queues/jobq.json) and every state change is appended to
sim_logs/jobs/events.jsonl (the control panel turns those into notifications).
"""
import fcntl, json, os, subprocess, sys, time

D = "sim_logs/jobs"
Q = D + "/queue.json"
EV = D + "/events.jsonl"
LOG = D + "/jobq.log"
PRESETS = {
    "eval150": ("./run_honest.sh {arm} 150 7200 \"{env}\" \"{desc}\"", "eval-150 (10007 + i*7919)"),
    "holdout150": ("SEED0=20011 SEEDSTEP=104729 ./run_honest.sh {arm} 150 7200 \"{env}\" \"{desc}\"", "holdout-150 (20011 + i*104729)"),
    "eval150_vs_K64": ("./run_honest.sh {arm} 150 7200 \"{env}\" \"{desc}\" && python3 ledger.py compare cpp_1ply_K64 {arm}",
                       "eval-150, then ledger compare vs mainline_h1 (cpp_1ply_K64)"),
}


def load():
    if not os.path.exists(Q):
        return {"next_id": 1, "jobs": []}
    return json.load(open(Q))


def save(q):
    os.makedirs(D, exist_ok=True)
    json.dump(q, open(Q + ".t", "w"), indent=1); os.replace(Q + ".t", Q)
    runs = [j["arm"] for j in q["jobs"] if j.get("arm") and j["state"] != "cancelled"]
    json.dump({"name": "Job queue", "runs": runs, "totals": {r: (500 if r.startswith("eval500") else 150) for r in runs},
               "est_secs": {r: 3600 for r in runs}}, open("sim_logs/queues/jobq.json", "w"), indent=1)


def locked(fn):
    os.makedirs(D, exist_ok=True)
    with open(D + "/.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        q = load()
        r = fn(q)
        save(q)
        return r


def event(job, state):
    with open(EV, "a") as f:
        f.write(json.dumps({"t": time.time(), "id": job["id"], "name": job["name"], "state": state}) + "\n")


def add(name, cmd, arm=None):
    def f(q):
        j = {"id": q["next_id"], "name": name, "cmd": cmd, "arm": arm, "state": "pending", "added": time.time()}
        q["next_id"] += 1
        q["jobs"].append(j)
        return j
    j = locked(f)
    print("added job %d: %s" % (j["id"], name))


def runner():
    lk = open(D + "/.runner", "w")
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("runner already active"); return
    while True:
        # never overlap the overnight queue
        pf = "sim_logs/overnight/runner.pid"
        if os.path.exists(pf):
            try:
                os.kill(int(open(pf).read()), 0)
                time.sleep(60); continue
            except (OSError, ValueError):
                pass

        def pick(q):
            for j in q["jobs"]:
                if j["state"] == "running":          # runner died mid-job: resume (run_honest skips finished seeds)
                    return j
            for j in q["jobs"]:
                if j["state"] == "pending":
                    j["state"] = "running"; j["started"] = time.time()
                    return j
            return None
        j = locked(pick)
        if j is None:                    # queue drained: daily digest
            subprocess.call(["nice", "-n", "10", ".venv/bin/python", "dash_data.py", "digest", "--write"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            event({"id": 0, "name": "queue drained - digest written"}, "done")
            return
        event(j, "running")
        with open(LOG, "a") as log:
            log.write("[%s] start job %d %s: %s\n" % (time.strftime("%F %T"), j["id"], j["name"], j["cmd"])); log.flush()
            rc = subprocess.call(["nice", "-n", "5", "bash", "-c", j["cmd"]], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            log.write("[%s] end job %d rc=%d\n" % (time.strftime("%F %T"), j["id"], rc))
        st = "done" if rc == 0 else "failed"

        def fin(q):
            for x in q["jobs"]:
                if x["id"] == j["id"] and x["state"] == "running":
                    x["state"] = st; x["rc"] = rc; x["finished"] = time.time()
        locked(fin)
        event(j, st)


def main(a):
    cmd = a[0] if a else "list"
    if cmd == "list":
        q = load()
        if "--json" in a:
            print(json.dumps(q)); return
        for j in q["jobs"]:
            print("%3d  %-9s %-28s %s" % (j["id"], j["state"], j["name"], j["cmd"][:90]))
    elif cmd == "presets":
        for k, (c, d) in PRESETS.items():
            print("%-16s %s\n                 %s" % (k, d, c))
    elif cmd == "add":
        i = a.index("--")
        add(a[1], " ".join(a[i + 1:]))
    elif cmd == "add-preset":
        p, arm, env = a[1], a[2], a[3]
        desc = a[4] if len(a) > 4 else env
        add("%s %s" % (p, arm), PRESETS[p][0].format(arm=arm, env=env, desc=desc), arm=arm)
        locked(lambda q: q["jobs"][-1].__setitem__("est_hours", {"eval150": 1.0, "holdout150": 0.8, "eval150_vs_K64": 1.0}.get(p, 1.0)))
    elif cmd == "est":                  # est <id> <hours>: approximate duration for the dashboard queue bars
        jid, hrs = int(a[1]), float(a[2])
        def f(q):
            for j in q["jobs"]:
                if j["id"] == jid:
                    j["est_hours"] = hrs
        locked(f)
    elif cmd in ("cancel", "retry", "move"):
        jid = int(a[1])

        def f(q):
            js = q["jobs"]
            k = next(i for i, j in enumerate(js) if j["id"] == jid)
            if cmd == "cancel" and js[k]["state"] == "pending":
                js[k]["state"] = "cancelled"
            elif cmd == "retry" and js[k]["state"] in ("failed", "cancelled"):
                js[k]["state"] = "pending"
            elif cmd == "move":
                t = k - 1 if a[2] == "up" else k + 1
                if 0 <= t < len(js) and js[k]["state"] == "pending" and js[t]["state"] == "pending":
                    js[k], js[t] = js[t], js[k]
        locked(f)
        main(["list"])
    elif cmd == "run":
        runner()
    elif cmd == "dash":                 # one JSON blob for the control panel
        since = int(a[1]) if len(a) > 1 else 0
        ov = []
        if os.path.exists("sim_logs/overnight/state.json"):
            ov = json.load(open("sim_logs/overnight/state.json"))["jobs"]
        ev = open(EV).read().splitlines() if os.path.exists(EV) else []
        print(json.dumps({"overnight": ov, "jobs": load()["jobs"], "presets": sorted(PRESETS),
                          "events": [json.loads(x) for x in ev[since:]], "events_n": len(ev)}))


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/sts"))
    main(sys.argv[1:])
