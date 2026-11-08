import json, glob, collections
rows = [f for p in glob.glob("sim_logs/headroom/seed_*.json") for f in json.load(open(p))["fights"]]
print("fights %d from %d seeds" % (len(rows), len(glob.glob("sim_logs/headroom/seed_*.json"))))
g = collections.defaultdict(list)
for r in rows:
    g[("A%d %s" % (r["act"], r["room"]))].append(r)
    g["ALL"].append(r)
print("%-10s %5s  %-22s %-22s %s" % ("type", "n", "honest win / HP lost", "clair win / HP lost", "gap (win pts, HP)"))
for k in sorted(g):
    v = g[k]; n = len(v)
    hw = sum(x["honest"]["won"] for x in v) / n; cw = sum(x["clair"]["won"] for x in v) / n
    hl = sum(x["hp_in"] - x["honest"]["hp_end"] for x in v) / n; cl = sum(x["hp_in"] - x["clair"]["hp_end"] for x in v) / n
    print("%-10s %5d  %5.1f%% / %5.1f        %5.1f%% / %5.1f        %+5.1f pts, %+5.1f HP" % (k, n, 100 * hw, hl, 100 * cw, cl, 100 * (cw - hw), cl - hl))
