"""Check the trainer's obs-derived deck metrics against a real worker.
Launches VecEnv(2) via the trainer's own code, prints deck/phi at reset
(expect ~10 cards, phi ~-1.7 for the starter deck) and after random steps,
plus a sample of the PBRS shaping terms.
"""
import numpy as np
from train_sim_mp import VecEnv, phi_of, deck_sizes_of, PHI_W, GAMMA

vec = VecEnv(2)
print("reset deck:", deck_sizes_of(vec.obs), "phi:", phi_of(vec.obs))
raw = vec.obs[0, 19:239] * 7.0
nz = np.nonzero(np.rint(raw))[0]
print("env0 reset deck slots:", [(int(i), int(round(raw[i]))) for i in nz])

rng = np.random.default_rng(3)
phi_prev = phi_of(vec.obs)
for t in range(400):
    acts = []
    for i in range(2):
        legal = np.nonzero(vec.mask[i])[0]
        acts.append(int(rng.choice(legal)))
    _, r, d, _, epfl = vec.step(np.array(acts))
    phi_new = phi_of(vec.obs)
    if t % 80 == 0 or any(d):
        term = np.where(d == 0.0, PHI_W * (GAMMA * phi_new - phi_prev), 0.0)
        print("t=%3d r=%s done=%s deck=%s phi=%s shaping=%s"
              % (t, np.round(r, 1), d, deck_sizes_of(vec.obs),
                 np.round(phi_new, 2), np.round(term, 2)))
    phi_prev = phi_new
vec.close()
print("probe done")
