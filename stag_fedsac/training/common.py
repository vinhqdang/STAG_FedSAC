"""
Shared episode machinery for every learning-based method and the rule-based
baselines: environment, history window, schedule tensor, and a single
evaluation routine so that all methods are scored identically on the same
held-out episodes (fixed seeds).
"""
from __future__ import annotations

import numpy as np
import torch

from stag_fedsac.config import *
from stag_fedsac.environment.schedule_generator import ScheduleGenerator
from stag_fedsac.environment.wifi_env import WiFiEnvironment

TEST_SEED_BASE = 1_000_000          # held-out episodes: seeds TEST_SEED_BASE + i


def make_env(n_aps: int, n_channels: int, seed: int, **kw) -> WiFiEnvironment:
    params = dict(
        n_aps=n_aps, n_users_max=N_USERS_MAX, n_channels=n_channels, p_min=P_MIN, p_max=P_MAX,
        timestep_s=TIMESTEP_S, episode_len=STEPS_PER_EPISODE, seed=seed,
        switch_outage_s=SWITCH_OUTAGE_S, edca_levels=EDCA_LEVELS, util_limit=UTIL_LIMIT,
        zone_assignments={i: i % N_ZONES for i in range(n_aps)},
    )
    params.update(kw)
    return WiFiEnvironment(**params)


class Harness:
    """Environment + history/schedule handling + common evaluation."""

    def __init__(self, n_aps: int, n_channels: int, device: torch.device, seed: int = 0,
                 env_kwargs: dict | None = None):
        self.n_aps, self.n_channels, self.device, self.seed = n_aps, n_channels, device, seed
        self.env_kwargs = env_kwargs or {}
        self.env = make_env(n_aps, n_channels, seed, **self.env_kwargs)
        self.zone_vec = np.array([self.env.zone_assignments[i] for i in range(n_aps)])
        self.sched_gen = ScheduleGenerator(n_aps, DELTA_HORIZON, D_SCHEDULE,
                                           zone_assignments=self.env.zone_assignments)
        self.A = self.env.get_adjacency_tensor().to(device)
        self._hist: list[np.ndarray] = []

    # -- history / schedule ---------------------------------------------------
    def start_episode(self, obs: dict) -> None:
        self._hist = [obs["h"]]

    def push_history(self, obs: dict) -> None:
        self._hist.append(obs["h"])
        if len(self._hist) > T_HISTORY:
            self._hist.pop(0)

    def history_array(self) -> np.ndarray:
        """[N, T, F]; short histories are left-padded with the oldest frame."""
        hist = list(self._hist)
        while len(hist) < T_HISTORY:
            hist.insert(0, hist[0])
        return np.stack(hist, axis=1).astype(np.float32)

    def schedule_array(self) -> np.ndarray:
        return self.sched_gen.generate_schedule_tensor(self.env.step_count, self.env.schedule_events)

    # -- surge labelling (for Surge-RMSE) ----------------------------------------
    def surge_flags(self, target_step: int) -> np.ndarray:
        """True for AP/target-step pairs that lie within one step of a large (>=100
        passengers) scheduled event of the AP's zone."""
        flags = np.zeros(self.n_aps, dtype=bool)
        for ev in self.env.schedule_events:
            if ev["n_passengers"] >= 100 and abs(ev["time_offset"] - target_step) <= 1.0:
                flags |= self.zone_vec == ev["zone"]
        return flags

    # -- hooks implemented by methods ------------------------------------------------
    def on_episode_start(self, obs: dict) -> None:           # pragma: no cover
        pass

    def decide(self, obs: dict, deterministic: bool = True):  # pragma: no cover
        """Return (actions [N, 1+C+3] float array, load forecast [N, delta] or None)."""
        raise NotImplementedError

    def after_step(self, next_obs: dict) -> None:
        self.push_history(next_obs)

    # -- evaluation --------------------------------------------------------------------
    def evaluate(self, n_episodes: int = 20, seed_base: int = TEST_SEED_BASE) -> list[dict]:
        out = []
        for i in range(n_episodes):
            obs = self.env.reset(seed=seed_base + i)
            self.start_episode(obs)
            self.on_episode_start(obs)
            rec = {k: [] for k in ("reward", "tput", "lat", "jfi", "qos", "eeff", "switch", "util")}
            loads = [obs["true_loads"].copy()]
            preds, surge = [], []
            for t in range(STEPS_PER_EPISODE):
                acts, lhat = self.decide(obs, deterministic=True)
                nxt, rewards, done, info = self.env.step({j: acts[j] for j in range(self.n_aps)})
                sys = info["sys"]
                rec["reward"].append(float(np.mean(list(rewards.values()))))
                rec["tput"].append(sys["mean_user_tput"])
                rec["lat"].append(sys["mean_user_lat"])
                rec["jfi"].append(sys["jfi_users"])
                rec["qos"].append(sys["qos_sat"])
                rec["eeff"].append(sys["energy_eff"])
                rec["switch"].append(sys["n_switch"] / self.n_aps)
                rec["util"].append(sys["mean_util"])
                loads.append(nxt["true_loads"].copy())
                preds.append(lhat)
                self.after_step(nxt)
                obs = nxt
            ep = {k: float(np.mean(v)) for k, v in rec.items()}
            ep.update(self._prediction_metrics(loads, preds))
            out.append(ep)
        return out

    def _prediction_metrics(self, loads: list, preds: list) -> dict:
        if preds[0] is None:
            return {}
        P, T_, M, S_ = [], [], [], []
        for t, lhat in enumerate(preds):
            for k in range(lhat.shape[1]):
                tgt_step = t + 1 + k
                if tgt_step <= STEPS_PER_EPISODE:
                    P.append(lhat[:, k]); T_.append(loads[tgt_step]); M.append(k)
                    S_.append(self._surge_at(tgt_step))
        P, T_, M, S_ = np.stack(P), np.stack(T_), np.array(M), np.stack(S_)
        err = P - T_
        res = {
            "rmse": float(np.sqrt((err ** 2).mean())),
            "mae": float(np.abs(err).mean()),
            "rmse_h1": float(np.sqrt((err[M == 0] ** 2).mean())),
            "surge_rmse": float(np.sqrt((err[S_] ** 2).mean())) if S_.any() else float("nan"),
            "normal_rmse": float(np.sqrt((err[~S_] ** 2).mean())) if (~S_).any() else float("nan"),
        }
        nz = T_ > 0.05
        res["mape"] = float((np.abs(err[nz]) / T_[nz]).mean() * 100) if nz.any() else float("nan")
        return res

    def _surge_at(self, target_step: int) -> np.ndarray:
        return self.surge_flags(target_step)


def summarise(episodes: list[dict]) -> dict:
    """Mean over evaluation episodes for each metric (NaN-aware)."""
    keys = episodes[0].keys()
    return {k: float(np.nanmean([e[k] for e in episodes])) for k in keys}
