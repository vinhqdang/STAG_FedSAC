"""
Run one (method, seed) experiment and write a JSON record with the training
history, per-episode test metrics under several evaluation conditions, and the
measured communication volume.

    python -m stag_fedsac.evaluation.run_experiment --method full --seed 0 --episodes 2000
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from stag_fedsac.config import *
from stag_fedsac.evaluation.baselines import DDPGBaseline, LLFBaseline, PersistenceForecaster, SSFBaseline
from stag_fedsac.training.common import make_env, summarise
from stag_fedsac.training.joint_trainer import JointTrainer

# method -> JointTrainer keyword arguments
SAC_METHODS = {
    "full":       dict(fed_mode="hier_kd"),
    "a1_nojoint": dict(fed_mode="hier_kd", use_joint_training=False),
    "a2_nosched": dict(fed_mode="hier_kd", use_schedule=False),
    "a3_noemb":   dict(fed_mode="hier_kd", use_embedding=False),
    "a4_zone":    dict(fed_mode="zone"),
    "a5_hieravg": dict(fed_mode="hier_avg"),
    "iso":        dict(fed_mode="none"),
    "flat":       dict(fed_mode="flat"),
    "central":    dict(fed_mode="central"),
}
ALL_METHODS = ["ssf", "llf", "persist", "lstm_drl", "fedddpg"] + list(SAC_METHODS)

# evaluation conditions: (MAC contention on, channel-switch outage in seconds)
CONDITIONS = {"main": (True, SWITCH_OUTAGE_S), "no_mac": (False, SWITCH_OUTAGE_S),
              "no_switch_cost": (True, 0.0), "switch_10s": (True, 10.0)}


def build(method: str, args) -> object:
    dev = DEVICE
    env_kwargs = dict(mac_contention=not args.train_no_mac, switch_outage_s=args.train_outage)
    if args.reward_weights:
        tp, lat, fair, qos = args.reward_weights
        env_kwargs["reward_weights"] = dict(tp=tp, lat=lat, fair=fair, qos=qos)
    if method == "ssf":
        return SSFBaseline(args.n_aps, N_CHANNELS, dev, args.seed, env_kwargs)
    if method == "persist":
        return PersistenceForecaster(args.n_aps, N_CHANNELS, dev, args.seed, env_kwargs)
    if method == "llf":
        return LLFBaseline(args.n_aps, N_CHANNELS, dev, args.seed, env_kwargs)
    if method == "fedddpg":
        return DDPGBaseline("fed", args.n_aps, N_CHANNELS, dev, args.seed, args.episodes, env_kwargs)
    if method == "lstm_drl":
        return DDPGBaseline("lstm", args.n_aps, N_CHANNELS, dev, args.seed, args.episodes, env_kwargs)
    kw = dict(SAC_METHODS[method])
    if args.eta is not None:
        kw["eta"] = args.eta
    return JointTrainer(args.n_aps, N_CHANNELS, dev, args.seed, total_episodes=args.episodes,
                        env_kwargs=env_kwargs, **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=ALL_METHODS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=TOTAL_EPISODES)
    ap.add_argument("--n-aps", type=int, default=N_AP)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-outage", type=float, default=SWITCH_OUTAGE_S, help="channel-switch outage seen in training (s)")
    ap.add_argument("--train-no-mac", action="store_true", help="train in a contention-free simulator")
    ap.add_argument("--reward-weights", type=float, nargs=4, metavar=("TP", "LAT", "FAIR", "QOS"))
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eta", type=float, default=None, help="joint-gradient weight (default: config ETA_JOINT)")
    args = ap.parse_args()

    t0 = time.time()
    agent = build(args.method, args)
    history = agent.train(args.episodes, eval_every=args.eval_every) if hasattr(agent, "train") else {}
    result = dict(method=args.method, seed=args.seed, episodes=args.episodes, n_aps=args.n_aps,
                  train_outage=args.train_outage, train_no_mac=args.train_no_mac,
                  reward_weights=args.reward_weights, history=history, tests={})
    for name, (mac, outage) in CONDITIONS.items():
        agent.env.mac_contention, agent.env.switch_outage_s = mac, outage
        eps = agent.evaluate(args.n_test)
        result["tests"][name] = dict(episodes=eps, summary=summarise(eps))
    agent.env.mac_contention, agent.env.switch_outage_s = not args.train_no_mac, args.train_outage
    result["wall_s"] = time.time() - t0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f)
    s = result["tests"]["main"]["summary"]
    print(f"DONE {args.method} seed={args.seed} reward={s['reward']:.3f} lat={s['lat']:.1f} jfi={s['jfi']:.3f} "
          f"qos={s['qos']:.3f} tput={s['tput']:.2f} ({result['wall_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
