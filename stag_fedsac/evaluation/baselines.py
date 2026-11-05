"""
Baselines.

Rule-based : SSF (static plan, strongest-signal association), LLF (reactive, load-aware).
Learned    : LSTM-DRL (scalar LSTM forecast + one shared DDPG agent, no federation,
             no schedule) and FedDDPG (per-AP DDPG agents, flat FedAvg, no forecast,
             no schedule). Both are trained with the SAME protocol as STAG-FedSAC:
             same environment, reward (+ Lagrangian), warm-up, number of episodes and
             update schedule, and they are evaluated on the same held-out episodes.
"""
from __future__ import annotations

import copy
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stag_fedsac.config import *
from stag_fedsac.models.graph_sac import BatchedDDPGActor, BatchedDDPGCritic, per_agent_clip_
from stag_fedsac.training.common import Harness, summarise
from stag_fedsac.training.replay_buffer import BatchedReplayBuffer

ACTION_DIM = 1 + N_CHANNELS + 3          # environment action: power, channel one-hot, airtime split
DDPG_ACTION_DIM = 1 + 1 + 3              # DDPG: power, channel index (continuous), 3 airtime logits


# ------------------------------------------------------------------ rule-based
class SSFBaseline(Harness):
    """Strongest Signal First: users keep their strongest AP; fixed channel plan, mid power,
    equal airtime split. Nothing adapts."""

    def decide(self, obs, deterministic=True):
        a = np.zeros((self.n_aps, ACTION_DIM))
        a[:, 0] = (P_MIN + P_MAX) / 2
        for i in range(self.n_aps):
            a[i, 1 + i % self.n_channels] = 1.0
        a[:, 1 + self.n_channels:] = 1.0 / 3
        return a, None


class PersistenceForecaster(SSFBaseline):
    """Forecast reference (not a controller): predicts every future load equal to the current load;
    control actions are those of SSF. Used only to put the learned predictors' RMSE in context."""

    def decide(self, obs, deterministic=True):
        a, _ = super().decide(obs, deterministic)
        return a, np.repeat(obs["true_loads"][:, None], DELTA_HORIZON, axis=1)


class LLFBaseline(Harness):
    """Least Loaded First (reactive): lowers power when the AP is overloaded, raises it when it is
    idle, splits airtime by the observed QoS mix. Reacts to the current state only."""

    def decide(self, obs, deterministic=True):
        a = np.zeros((self.n_aps, ACTION_DIM))
        util = obs["util"]
        for i in range(self.n_aps):
            if util[i] > 0.8:
                a[i, 0] = P_MIN + (P_MAX - P_MIN) * 0.3
            elif util[i] < 0.3:
                a[i, 0] = P_MAX
            else:
                a[i, 0] = P_MIN + (P_MAX - P_MIN) * (1 - min(util[i], 1.0))
            a[i, 1 + i % self.n_channels] = 1.0
            bw = np.maximum(obs["qos"][i], 0.1)
            a[i, 1 + self.n_channels:] = bw / bw.sum()
        return a, None


# ------------------------------------------------------------------ learned
class LSTMPredictor(nn.Module):
    """Temporal-only load forecaster for each AP (no graph, no schedule): history -> next delta loads."""

    def __init__(self, input_dim: int = F_FEATURES, hidden: int = 128, layers: int = 2, delta: int = DELTA_HORIZON):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, delta), nn.Sigmoid())

    def forward(self, x):                      # [B, T, F] -> [B, delta]
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


class DDPGBaseline(Harness):
    """mode='fed' : FedDDPG   - N per-AP DDPG agents, flat FedAvg every T_local, no forecast.
       mode='lstm': LSTM-DRL  - ONE shared DDPG agent (parameter sharing, pooled data, no federation)
                    whose state is augmented with a scalar next-step load forecast from an LSTM."""

    def __init__(self, mode: str, n_aps: int = N_AP, n_channels: int = N_CHANNELS,
                 device: torch.device = DEVICE, seed: int = 0, total_episodes: int = TOTAL_EPISODES,
                 env_kwargs: dict | None = None):
        torch.manual_seed(seed)
        np.random.seed(seed)
        super().__init__(n_aps, n_channels, device, seed, env_kwargs)
        assert mode in ("fed", "lstm")
        self.mode, self.total_episodes = mode, total_episodes
        self.n_agents = n_aps if mode == "fed" else 1
        self.state_dim = F_FEATURES + 3 + RADIO_DIM + (1 if mode == "lstm" else 0)
        self.actor = BatchedDDPGActor(self.n_agents, self.state_dim, DDPG_ACTION_DIM).to(device)
        self.critic = BatchedDDPGCritic(self.n_agents, self.state_dim, DDPG_ACTION_DIM).to(device)
        if mode == "fed":
            self.actor.init_default(torch.arange(n_aps, device=device) % n_channels, n_channels)
        else:                                   # shared agent: zero channel output = incumbent plan (relative action)
            self.actor.init_default(torch.full((1,), (n_channels - 1) // 2, device=device), n_channels)
        self.t_actor = copy.deepcopy(self.actor).requires_grad_(False)
        self.t_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        cap = min(REPLAY_BUFFER_SIZE * (1 if mode == "fed" else N_AP), total_episodes * STEPS_PER_EPISODE * (1 if mode == "fed" else n_aps) + 8)
        self.buffer = BatchedReplayBuffer(self.n_agents, cap, self.state_dim, DDPG_ACTION_DIM, device)
        self.lambdas = torch.zeros(n_aps, device=device)
        if mode == "lstm":
            self.lstm = LSTMPredictor().to(device)
            self.opt_l = torch.optim.Adam(self.lstm.parameters(), lr=LR_PREDICTOR)
            self._lstm_buf: list = []
        self.global_step = 0
        self.noise = 0.2

    # state: [N, S] (LSTM-DRL uses the same state layout for all APs; a single agent acts for every AP)
    def _states(self, obs: dict, lhat: torch.Tensor | None) -> torch.Tensor:
        h = torch.as_tensor(obs["h"], device=self.device)
        q = torch.as_tensor(obs["qos"], device=self.device)
        parts = [h, q, torch.as_tensor(obs["radio"], device=self.device)] + ([lhat[:, :1]] if self.mode == "lstm" else [])
        return torch.cat(parts, dim=-1)

    def _forecast(self):
        if self.mode != "lstm":
            return None
        with torch.no_grad():
            return self.lstm(torch.as_tensor(self.history_array(), device=self.device))

    def _to_env(self, a: torch.Tensor) -> np.ndarray:
        """DDPG action in [-1,1]^5 -> physical action. The channel is a single continuous output that is
        rounded to an index (the usual way of using DDPG for a discrete choice)."""
        a = a.clamp(-1, 1)
        out = torch.zeros(a.shape[0], ACTION_DIM, device=a.device)
        out[:, 0] = (a[:, 0] + 1) / 2 * (P_MAX - P_MIN) + P_MIN
        if self.mode == "fed":
            ch = torch.round((a[:, 1] + 1) / 2 * (self.n_channels - 1)).long()
        else:
            # the shared agent has no AP-specific parameters: its channel output is an offset (in channels)
            # from the AP's incumbent channel, so that a zero output reproduces the incumbent plan
            default = torch.arange(a.shape[0], device=a.device) % self.n_channels
            ch = (default + torch.round(a[:, 1] * (self.n_channels - 1) / 2).long()) % self.n_channels
        out[torch.arange(a.shape[0], device=a.device), 1 + ch] = 1.0
        out[:, 1 + self.n_channels:] = F.softmax(3.0 * a[:, 2:5], dim=-1)
        return out.cpu().numpy()

    def _act(self, s: torch.Tensor, explore: bool) -> torch.Tensor:
        with torch.no_grad():
            if self.mode == "fed":
                a = self.actor(s.unsqueeze(1))[:, 0]
            else:
                a = self.actor(s.unsqueeze(0))[0]          # [N, A] from the shared actor
            if explore:
                a = (a + self.noise * torch.randn_like(a)).clamp(-1, 1)
        return a

    def decide(self, obs, deterministic=True):
        lhat = self._forecast()
        s = self._states(obs, lhat)
        a = self._act(s, explore=False)
        return self._to_env(a), (lhat.cpu().numpy() if lhat is not None else None)

    def _update(self) -> None:
        s, a, r, s2, d = self.buffer.sample(BATCH_SIZE)
        with torch.no_grad():
            y = r + GAMMA_DISCOUNT * (1 - d) * self.t_critic(s2, self.t_actor(s2))
        closs = (self.critic(s, a) - y).pow(2).mean(dim=(1, 2)).sum()
        self.opt_c.zero_grad(set_to_none=True)
        closs.backward()
        per_agent_clip_(list(self.critic.parameters()), 1.0)
        self.opt_c.step()
        self.critic.requires_grad_(False)
        a_pi = self.actor(s)
        aloss = -self.critic(s, a_pi).mean(dim=(1, 2)).sum()
        if BC_WEIGHT > 0:
            aloss = aloss + BC_WEIGHT * (a_pi - self._ref_action()).pow(2).sum(-1).mean(dim=1).sum()
        self.opt_a.zero_grad(set_to_none=True)
        aloss.backward()
        per_agent_clip_(list(self.actor.parameters()), 1.0)
        self.opt_a.step()
        self.critic.requires_grad_(True)
        with torch.no_grad():
            for t, p in ((self.t_actor, self.actor), (self.t_critic, self.critic)):
                tp, cp = list(t.parameters()), list(p.parameters())
                torch._foreach_mul_(tp, 1 - TAU_SOFT)
                torch._foreach_add_(tp, cp, alpha=TAU_SOFT)

    def _ref_action(self) -> torch.Tensor:
        """Incumbent configuration in the DDPG action space: mid power, incumbent channel, equal airtime."""
        if not hasattr(self, "_a_ref"):
            n = self.n_agents
            ref = torch.zeros(n, 1, DDPG_ACTION_DIM, device=self.device)
            ref[:, 0, 0] = 2 * (DEFAULT_POWER_DBM - P_MIN) / (P_MAX - P_MIN) - 1
            plan = torch.arange(n, device=self.device) % self.n_channels if self.mode == "fed" else torch.full((n,), (self.n_channels - 1) // 2, device=self.device)
            ref[:, 0, 1] = 2.0 * plan.float() / (self.n_channels - 1) - 1.0 if self.mode == "fed" else 0.0
            self._a_ref = ref
        return self._a_ref

    def _lstm_update(self) -> None:
        if len(self._lstm_buf) < 8:
            return
        idx = np.random.randint(0, len(self._lstm_buf), PRED_BATCH)
        H = torch.as_tensor(np.stack([self._lstm_buf[i][0] for i in idx]), device=self.device)      # [B,N,T,F]
        tgt = torch.as_tensor(np.stack([self._lstm_buf[i][1] for i in idx]), device=self.device)
        mask = torch.as_tensor(np.stack([self._lstm_buf[i][2] for i in idx]), device=self.device).unsqueeze(1)
        pred = self.lstm(H.flatten(0, 1)).view(len(idx), self.n_aps, DELTA_HORIZON)
        loss = ((pred - tgt).pow(2) * mask).sum() / (mask.sum() * self.n_aps)
        self.opt_l.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_l.step()

    def train(self, total_episodes: int | None = None, eval_every: int = 100, eval_eps: int = 5,
              verbose: bool = True) -> dict:
        total_episodes = total_episodes or self.total_episodes
        hist = defaultdict(list)
        t_start = time.time()
        for ep in range(total_episodes):
            obs = self.env.reset()
            self.start_episode(obs)
            loads = [obs["true_loads"].copy()]
            samples, ep_env_r = [], []
            for t in range(STEPS_PER_EPISODE):
                self.global_step += 1
                lhat = self._forecast()
                s = self._states(obs, lhat)
                if self.global_step <= WARMUP_STEPS:
                    a = self._act(s, explore=True)       # warm-up: noisy incumbent configuration
                else:
                    a = self._act(s, explore=True)
                nxt, rewards, done, info = self.env.step({i: self._to_env(a)[i] for i in range(self.n_aps)})
                samples.append(self.history_array())
                self.push_history(nxt)
                lhat2 = self._forecast()
                s2 = self._states(nxt, lhat2)
                env_r = torch.as_tensor([rewards[i] for i in range(self.n_aps)], dtype=torch.float32, device=self.device)
                util = torch.as_tensor(nxt["util"], device=self.device)
                viol = (util - UTIL_LIMIT).clamp(min=0, max=UTIL_VIOL_CLIP)
                total = R_SCALE * (env_r - self.lambdas * viol)
                self.lambdas = (self.lambdas + RHO_LAGRANGIAN * (util - UTIL_LIMIT).clamp(max=UTIL_VIOL_CLIP)).clamp(0, LAMBDA_MAX)
                if self.mode == "fed":
                    self.buffer.add(s, a, total, s2, float(done))
                else:
                    self._pooled_add(s, a, total, s2, float(done))
                loads.append(nxt["true_loads"].copy())
                ep_env_r.append(float(env_r.mean()))
                if self.global_step % UPDATE_FREQ == 0 and self.global_step > WARMUP_STEPS and len(self.buffer) >= BATCH_SIZE:
                    self._update()
                if self.mode == "lstm" and self.global_step % UPDATE_FREQ == 0:
                    self._lstm_update()
                if self.mode == "fed" and self.global_step % T_LOCAL_FED == 0:
                    self._fedavg()
                obs = nxt
            if self.mode == "lstm":
                for t, H in enumerate(samples):
                    tgt = np.zeros((self.n_aps, DELTA_HORIZON), dtype=np.float32)
                    mask = np.zeros(DELTA_HORIZON, dtype=np.float32)
                    for k in range(DELTA_HORIZON):
                        if t + 1 + k <= STEPS_PER_EPISODE:
                            tgt[:, k] = loads[t + 1 + k]; mask[k] = 1
                    self._lstm_buf.append((H, tgt, mask))
                self._lstm_buf = self._lstm_buf[-PRED_BUFFER:]
            hist["train_env_reward"].append(float(np.mean(ep_env_r)))
            if (ep + 1) % eval_every == 0 or ep == 0:
                m = summarise(self.evaluate(eval_eps, seed_base=500_000))
                hist["eval_episode"].append(ep + 1)
                hist["eval_reward"].append(m["reward"])
                if verbose:
                    print(f"[{'FedDDPG' if self.mode=='fed' else 'LSTM-DRL'}] ep {ep+1:5d} train {hist['train_env_reward'][-1]:.3f} "
                          f"eval {m['reward']:.3f} lat {m['lat']:.0f}ms ({time.time()-t_start:.0f}s)", flush=True)
        hist = dict(hist)
        hist["train_time_s"] = time.time() - t_start
        return hist

    def _pooled_add(self, s, a, r, s2, d) -> None:
        """LSTM-DRL: every AP's transition is stored in the single shared agent's buffer."""
        b = self.buffer
        for i in range(self.n_aps):
            j = b.ptr
            b.s[0, j] = s[i].half(); b.s2[0, j] = s2[i].half(); b.a[0, j] = a[i]
            b.r[0, j, 0] = r[i]; b.d[0, j, 0] = d
            b.ptr = (b.ptr + 1) % b.cap
            b.size = min(b.size + 1, b.cap)

    @torch.no_grad()
    def _fedavg(self) -> None:
        """Flat FedAvg over all APs (single level). The actor's output layer ``l3`` stays personal, as in
        STAG-FedSAC (otherwise every AP is forced onto the same channel); critics are averaged completely."""
        for p in list(self.actor.l1.parameters()) + list(self.actor.l2.parameters()) + list(self.critic.parameters()):
            p.data[:] = p.data.mean(dim=0, keepdim=True).expand_as(p.data)
