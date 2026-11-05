"""
Headroom reference: a greedy per-interval optimiser with perfect knowledge of the users present in the
interval (it evaluates candidate configurations in the simulator itself) and no switching cost. It is not
a deployable controller; it indicates how much reward any controller could gain over the static plan.

    python analysis/oracle_headroom.py --episodes 10 --out results/oracle.json
"""
import argparse, json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from stag_fedsac.config import *
from stag_fedsac.training.common import make_env, TEST_SEED_BASE
from stag_fedsac.evaluation.baselines import SSFBaseline

ap = argparse.ArgumentParser(); ap.add_argument("--episodes", type=int, default=10); ap.add_argument("--out", default="results/oracle.json")
ap.add_argument("--sweeps", type=int, default=2); a = ap.parse_args()
C = N_CHANNELS
env = make_env(N_AP, C, seed=0)


def score():
    net = env._compute_network()
    return float(np.mean([env._compute_reward(i, net)[0] for i in range(N_AP)]))


def episode(seed):
    env.reset(seed=seed)
    ch = np.arange(N_AP) % C; pw = np.full(N_AP, DEFAULT_POWER_DBM); bw = np.full((N_AP, 3), 1 / 3)
    out = []
    for t in range(STEPS_PER_EPISODE):
        env._simulate_mobility(); env._apply_schedule_effects(); env.step_count += 1
        env.switched[:] = False

        def apply():
            env.ap_channels = ch.copy(); env.ap_powers = pw.copy(); env.ap_bw_alloc = bw.copy()
        apply(); best = score()
        for _ in range(a.sweeps):
            for i in range(N_AP):
                for c in range(C):
                    if c == ch[i]: continue
                    old = ch[i]; ch[i] = c; apply(); s = score()
                    if s > best + 1e-9: best = s
                    else: ch[i] = old
                for p in (8, 11, 14, 17, 20):
                    old = pw[i]; pw[i] = p; apply(); s = score()
                    if s > best + 1e-9: best = s
                    else: pw[i] = old
                for b in ([.0625, .5, .4375], [.2, .4, .4], [1 / 3] * 3, [.1, .3, .6]):
                    old = bw[i].copy(); bw[i] = b; apply(); s = score()
                    if s > best + 1e-9: best = s
                    else: bw[i] = old
        apply(); out.append(score())
    return float(np.mean(out))


res = [episode(TEST_SEED_BASE + i) for i in range(a.episodes)]
ssf = SSFBaseline(N_AP, C, torch.device("cpu"), 0, {})
ssf_res = [e["reward"] for e in ssf.evaluate(a.episodes)]
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(dict(oracle=res, ssf=ssf_res, oracle_mean=float(np.mean(res)), ssf_mean=float(np.mean(ssf_res)), episodes=a.episodes), open(a.out, "w"))
print("oracle", np.mean(res), "ssf", np.mean(ssf_res))
