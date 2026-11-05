"""
STAG-FedSAC trainer.

Per-AP Soft Actor-Critic agents (stacked/batched), the ST-GCAT predictor with an
optional bidirectional joint gradient, the airtime-utilisation Lagrangian, and
the federation modes of ``hierfed_kd``.
"""
from __future__ import annotations

import copy
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from stag_fedsac.config import *
from stag_fedsac.models.graph_sac import BatchedActor, BatchedCritic, per_agent_clip_
from stag_fedsac.models.hierfed_kd import Federation
from stag_fedsac.models.stgcat import STGCAT
from stag_fedsac.training.common import Harness, summarise
from stag_fedsac.training.replay_buffer import BatchedReplayBuffer

STATE_DIM = F_FEATURES + DELTA_HORIZON * D_HIDDEN + 3 + D_SCHEDULE + RADIO_DIM      # 797
ACTION_DIM = 1 + N_CHANNELS + 3


class JointTrainer(Harness):
    """fed_mode in {none, flat, zone, hier_avg, hier_kd, central}.

    central: no weight sharing, but every agent samples its minibatches from the
    pooled experience of all APs (privacy-violating upper bound used as a comparator).
    """

    def __init__(self, n_aps: int = N_AP, n_channels: int = N_CHANNELS,
                 device: torch.device = DEVICE, seed: int = 0, fed_mode: str = "hier_kd",
                 use_joint_training: bool = True, use_schedule: bool = True,
                 use_embedding: bool = True, eta: float = ETA_JOINT,
                 total_episodes: int = TOTAL_EPISODES, env_kwargs: dict | None = None,
                 kd_steps: int = 20, log_path: str | None = None):
        torch.manual_seed(seed)
        np.random.seed(seed)
        super().__init__(n_aps, n_channels, device, seed, env_kwargs)
        self.fed_mode, self.eta = fed_mode, eta
        self.use_joint, self.use_schedule, self.use_embedding = use_joint_training, use_schedule, use_embedding
        self.total_episodes = total_episodes
        self.log_path = log_path

        self.stgcat = STGCAT(F_FEATURES, D_SCHEDULE, D_HIDDEN, N_HEADS, T_HISTORY,
                             DELTA_HORIZON, N_TRANSFORMER_LAYERS, DROPOUT).to(device)
        self.actor = BatchedActor(n_aps, STATE_DIM, n_channels, p_min=P_MIN, p_max=P_MAX).to(device)
        self.critic = BatchedCritic(n_aps, STATE_DIM, ACTION_DIM).to(device)
        self.actor.init_default(torch.arange(n_aps, device=device) % n_channels)   # incumbent plan
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = torch.full((1,), float(np.log(ALPHA_INIT)), requires_grad=True, device=device)
        self.target_entropy = TARGET_ENTROPY

        self.opt_pred = torch.optim.Adam(self.stgcat.parameters(), lr=LR_PREDICTOR)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=LR_ACTOR)

        mode = "none" if fed_mode == "central" else fed_mode
        self.fed = Federation(self.zone_vec, mode, FEDPROX_MU, LR_FEDERATED_KD, kd_steps,
                              n_channels, S_REF_SIZE, device=device)
        cap = min(REPLAY_BUFFER_SIZE, total_episodes * STEPS_PER_EPISODE + 8)
        self.buffer = BatchedReplayBuffer(n_aps, cap, STATE_DIM, ACTION_DIM, device)
        self._init_pred_buffer()
        self.lambdas = torch.zeros(n_aps, device=device)
        self.global_step = 0
        self.s_ref: torch.Tensor | None = None
        self.proxy_env = None
        self.tau = GUMBEL_TAU_START
        self.stgcat.eval()

    # ------------------------------------------------------------------ state
    def _build_states(self, obs: dict, Z: torch.Tensor, S: np.ndarray) -> torch.Tensor:
        """[N, 797] = [h_i (5), vec(Z_i) (768), q_i (3), s_i^sched (8), radio_i (13)] with ablation switches."""
        h = torch.as_tensor(obs["h"], device=self.device)
        z = EMB_SCALE * Z[0].flatten(1) if self.use_embedding else torch.zeros(self.n_aps, DELTA_HORIZON * D_HIDDEN, device=self.device)
        q = torch.as_tensor(obs["qos"], device=self.device)
        sc = torch.as_tensor(S[:, 0, :], device=self.device) if self.use_schedule else torch.zeros(self.n_aps, D_SCHEDULE, device=self.device)
        rad = torch.as_tensor(obs["radio"], device=self.device)
        return torch.cat([h, z, q, sc, rad], dim=-1)

    def _predict(self, H: np.ndarray, S: np.ndarray):
        S_in = S if self.use_schedule else np.zeros_like(S)
        with torch.no_grad():
            Z, Lhat = self.stgcat(torch.as_tensor(H, device=self.device).unsqueeze(0),
                                  torch.as_tensor(S_in, device=self.device).unsqueeze(0), self.A)
        return Z, Lhat

    # ------------------------------------------------------- predictor buffers
    def _init_pred_buffer(self) -> None:
        n, d = self.n_aps, self.device
        c = PRED_BUFFER
        self.pb = dict(
            H=torch.zeros(c, n, T_HISTORY, F_FEATURES, dtype=torch.float16, device=d),
            S=torch.zeros(c, n, DELTA_HORIZON, D_SCHEDULE, dtype=torch.float16, device=d),
            h=torch.zeros(c, n, F_FEATURES, dtype=torch.float16, device=d),
            q=torch.zeros(c, n, 3, dtype=torch.float16, device=d),
            rad=torch.zeros(c, n, RADIO_DIM, dtype=torch.float16, device=d),
            tgt=torch.zeros(c, n, DELTA_HORIZON, device=d),
            mask=torch.zeros(c, DELTA_HORIZON, device=d),
        )
        self.pb_ptr, self.pb_size = 0, 0
        self._ep_pred: list[dict] = []

    def _pred_commit_episode(self, loads: list[np.ndarray]) -> None:
        """Attach true multi-step targets to the samples collected in the episode."""
        for t, smp in enumerate(self._ep_pred):
            tgt = np.zeros((self.n_aps, DELTA_HORIZON), dtype=np.float32)
            mask = np.zeros(DELTA_HORIZON, dtype=np.float32)
            for k in range(DELTA_HORIZON):
                if t + 1 + k <= STEPS_PER_EPISODE:
                    tgt[:, k] = loads[t + 1 + k]
                    mask[k] = 1.0
            i = self.pb_ptr
            self.pb["H"][i] = torch.as_tensor(smp["H"], device=self.device).half()
            self.pb["S"][i] = torch.as_tensor(smp["S"], device=self.device).half()
            self.pb["h"][i] = torch.as_tensor(smp["h"], device=self.device).half()
            self.pb["q"][i] = torch.as_tensor(smp["q"], device=self.device).half()
            self.pb["rad"][i] = torch.as_tensor(smp["rad"], device=self.device).half()
            self.pb["tgt"][i] = torch.as_tensor(tgt, device=self.device)
            self.pb["mask"][i] = torch.as_tensor(mask, device=self.device)
            self.pb_ptr = (self.pb_ptr + 1) % PRED_BUFFER
            self.pb_size = min(self.pb_size + 1, PRED_BUFFER)
        self._ep_pred = []

    # ----------------------------------------------------------------- acting
    def decide(self, obs: dict, deterministic: bool = True):
        H, S = self.history_array(), self.schedule_array()
        Z, Lhat = self._predict(H, S)
        s = self._build_states(obs, Z, S)
        with torch.no_grad():
            a, _, _, _ = self.actor(s.unsqueeze(1), deterministic=deterministic, gumbel_tau=self.tau)
        return a[:, 0].cpu().numpy(), Lhat[0].cpu().numpy()

    # ----------------------------------------------------------- reference set
    def _build_reference_set(self) -> torch.Tensor:
        """Public reference states: one rollout of an independent proxy venue with
        random actions, observed through the current predictor; M states are sampled
        from all APs and steps. No AP's own replay data is used."""
        from stag_fedsac.training.common import make_env
        if self.proxy_env is None:
            self.proxy_env = make_env(self.n_aps, self.n_channels, self.seed + 7919, **self.env_kwargs)
        env = self.proxy_env
        obs = env.reset()
        saved = (list(self._hist), env.step_count)
        self._hist = [obs["h"]]
        states = []
        rng = np.random.default_rng(self.seed + self.global_step)
        for _ in range(STEPS_PER_EPISODE):
            H, S = self.history_array(), self.sched_gen.generate_schedule_tensor(env.step_count, env.schedule_events)
            Z, _ = self._predict(H, S)
            states.append(self._build_states(obs, Z, S))
            acts = {}
            for i in range(self.n_aps):
                v = np.zeros(ACTION_DIM)
                v[0] = rng.uniform(P_MIN, P_MAX)
                v[1 + rng.integers(self.n_channels)] = 1.0
                v[1 + self.n_channels:] = rng.dirichlet([1, 1, 1])
                acts[i] = v
            obs, _, _, _ = env.step(acts)
            self._hist.append(obs["h"])
            self._hist = self._hist[-T_HISTORY:]
        self._hist = saved[0]
        allst = torch.cat(states, dim=0)
        idx = torch.randperm(allst.shape[0], device=self.device)[:S_REF_SIZE]
        return allst[idx]

    # ------------------------------------------------------------------ updates
    def _sac_update(self) -> None:
        pooled = self.fed_mode == "central"
        s, a, r, s2, d = self.buffer.sample(BATCH_SIZE, pooled=pooled)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, lp2, _, _ = self.actor(s2, gumbel_tau=self.tau)
            q1t, q2t = self.target_critic(s2, a2)
            y = r + GAMMA_DISCOUNT * (1 - d) * (torch.min(q1t, q2t) - alpha * lp2)
        q1, q2 = self.critic(s, a)
        closs = 0.5 * ((q1 - y).pow(2).mean(dim=(1, 2)) + (q2 - y).pow(2).mean(dim=(1, 2))).sum()
        self.opt_critic.zero_grad(set_to_none=True)
        closs.backward()
        per_agent_clip_(list(self.critic.parameters()), 1.0)
        self.opt_critic.step()

        self.critic.requires_grad_(False)
        ac, lp, _, _ = self.actor(s, gumbel_tau=self.tau)
        q1c, q2c = self.critic(s, ac)
        aloss = (alpha * lp - torch.min(q1c, q2c)).mean(dim=(1, 2)).sum() + self.fed.prox_loss(self.actor)
        if BC_WEIGHT > 0:
            aloss = aloss + BC_WEIGHT * self._incumbent_distance(ac)
        self.opt_actor.zero_grad(set_to_none=True)
        aloss.backward()
        per_agent_clip_(list(self.actor.parameters()), 1.0)
        self.opt_actor.step()
        self.critic.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (lp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.opt_alpha.step()

        with torch.no_grad():
            tp, cp = list(self.target_critic.parameters()), list(self.critic.parameters())
            torch._foreach_mul_(tp, 1 - TAU_SOFT)
            torch._foreach_add_(tp, cp, alpha=TAU_SOFT)

    def _incumbent_distance(self, a: torch.Tensor) -> torch.Tensor:
        """Squared distance of sampled actions [N,B,A] from the incumbent configuration (power scaled to [0,1]);
        summed over agents, averaged over the batch. Acts as a prior for safe policy improvement."""
        if not hasattr(self, "_a_ref"):
            ref = torch.zeros(self.n_aps, 1, ACTION_DIM, device=self.device)
            ref[:, 0, 0] = (DEFAULT_POWER_DBM - P_MIN) / (P_MAX - P_MIN)
            ref[torch.arange(self.n_aps), 0, 1 + (torch.arange(self.n_aps) % self.n_channels)] = 1.0
            ref[:, 0, 1 + self.n_channels:] = 1.0 / 3
            self._a_ref = ref
        a_n = torch.cat([((a[..., :1] - P_MIN) / (P_MAX - P_MIN)), a[..., 1:]], dim=-1)
        return (a_n - self._a_ref).pow(2).sum(-1).mean(dim=1).sum()

    def _pred_update(self, with_actor: bool) -> float:
        """Predictor step on a minibatch of stored (history, schedule) samples:
        L_pred = MSE(L_hat, L) + eta * L_actor  (eta = 0 when the bridge is off)."""
        if self.pb_size < PRED_BATCH:
            return float("nan")
        idx = torch.randint(0, self.pb_size, (PRED_BATCH,), device=self.device)
        H, S = self.pb["H"][idx].float(), self.pb["S"][idx].float()
        tgt, mask = self.pb["tgt"][idx], self.pb["mask"][idx].unsqueeze(1)
        S_in = S if self.use_schedule else torch.zeros_like(S)
        self.stgcat.train()
        Z, Lhat = self.stgcat(H, S_in, self.A)                       # [B,N,delta,d_h], [B,N,delta]
        mse = ((Lhat - tgt).pow(2) * mask).sum() / (mask.sum() * self.n_aps)
        loss = mse
        if with_actor and self.use_joint and self.use_embedding and self.eta > 0:
            B = H.shape[0]
            z = EMB_SCALE * Z.flatten(2).permute(1, 0, 2)             # [N,B,delta*d_h]
            h = self.pb["h"][idx].float().permute(1, 0, 2)
            q = self.pb["q"][idx].float().permute(1, 0, 2)
            sc = (S[:, :, 0, :] if self.use_schedule else torch.zeros_like(S[:, :, 0, :])).permute(1, 0, 2)
            rad = self.pb["rad"][idx].float().permute(1, 0, 2)
            s_fresh = torch.cat([h, z, q, sc, rad], dim=-1)           # [N,B,797]
            a_f, lp_f, _, _ = self.actor(s_fresh, gumbel_tau=self.tau)
            q1, q2 = self.critic(s_fresh, a_f)
            qmin = torch.min(q1, q2)
            actor_term = (self.log_alpha.exp().detach() * lp_f - qmin).mean()
            # scale-invariant weighting: eta is relative to the critic's value scale (E|Q|), so the
            # bridge does not depend on the reward scaling
            actor_term = actor_term / (qmin.detach().abs().mean() + 1.0)
            loss = mse + self.eta * actor_term
        params = list(self.stgcat.parameters())
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        for p, g in zip(params, grads):
            p.grad = g
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        self.opt_pred.step()
        self.opt_pred.zero_grad(set_to_none=True)
        self.stgcat.eval()
        return float(mse.detach())

    # -------------------------------------------------------------------- train
    def train(self, total_episodes: int | None = None, eval_every: int = 100, eval_eps: int = 5,
              verbose: bool = True) -> dict:
        total_episodes = total_episodes or self.total_episodes
        hist = defaultdict(list)
        t_start = time.time()
        for ep in range(total_episodes):
            self.tau = max(GUMBEL_TAU_END, GUMBEL_TAU_START - (GUMBEL_TAU_START - GUMBEL_TAU_END) * ep / max(1, total_episodes - 1))
            obs = self.env.reset()
            self.start_episode(obs)
            loads = [obs["true_loads"].copy()]
            ep_env_r, ep_tot_r, ep_sw = [], [], []
            for t in range(STEPS_PER_EPISODE):
                self.global_step += 1
                H, S = self.history_array(), self.schedule_array()
                Z, Lhat = self._predict(H, S)
                s = self._build_states(obs, Z, S)
                with torch.no_grad():
                    # warm-up explores around the incumbent configuration (the actor starts there);
                    # afterwards the learned stochastic policy acts
                    act, _, _, _ = self.actor(s.unsqueeze(1), gumbel_tau=self.tau)
                    act = act[:, 0]
                a_np = act.cpu().numpy()
                nxt, rewards, done, info = self.env.step({i: a_np[i] for i in range(self.n_aps)})
                self.push_history(nxt)
                H2, S2 = self.history_array(), self.schedule_array()
                Z2, _ = self._predict(H2, S2)
                s2 = self._build_states(nxt, Z2, S2)

                env_r = torch.as_tensor([rewards[i] for i in range(self.n_aps)], dtype=torch.float32, device=self.device)
                util = torch.as_tensor(nxt["util"], device=self.device)
                viol = (util - UTIL_LIMIT).clamp(min=0, max=UTIL_VIOL_CLIP)
                bonus = -(Lhat[0, :, 0] - torch.as_tensor(nxt["true_loads"], device=self.device)).abs()
                penalty = -self.lambdas * viol
                total = R_SCALE * (env_r + DELTA_REWARD * bonus + penalty)
                self.buffer.add(s, act, total, s2, float(done))
                self.fed.update_reward(env_r)
                self.lambdas = (self.lambdas + RHO_LAGRANGIAN * (util - UTIL_LIMIT).clamp(max=UTIL_VIOL_CLIP)).clamp(0, LAMBDA_MAX)

                self._ep_pred.append(dict(H=H, S=S, h=obs["h"], q=obs["qos"], rad=obs["radio"]))
                loads.append(nxt["true_loads"].copy())
                ep_env_r.append(float(env_r.mean())); ep_tot_r.append(float(total.mean()))
                ep_sw.append(info["sys"]["n_switch"] / self.n_aps)

                # supervised predictor update runs from the start (also during warm-up)
                if self.global_step % UPDATE_FREQ == 0:
                    learning = self.global_step > WARMUP_STEPS and len(self.buffer) >= BATCH_SIZE
                    if learning:
                        self._sac_update()
                    self._pred_update(with_actor=learning)

                if self.global_step % T_LOCAL_FED == 0 and self.fed.mode != "none":
                    self.fed.zone_round(self.actor)
                if self.global_step % T_GLOBAL_FED == 0 and self.fed.mode in ("hier_avg", "hier_kd"):
                    if self.fed.mode == "hier_kd" and self.global_step > WARMUP_STEPS:
                        if self.s_ref is None or (self.global_step // T_GLOBAL_FED) % 5 == 0:
                            self.s_ref = self._build_reference_set()
                            self.fed.ship_reference_set(self.s_ref)
                        self.fed.global_round_kd(self.actor, self.s_ref)
                    elif self.fed.mode == "hier_avg":
                        self.fed.global_round_avg(self.actor)
                obs = nxt
            self._pred_commit_episode(loads)
            hist["train_env_reward"].append(float(np.mean(ep_env_r)))
            hist["train_total_reward"].append(float(np.mean(ep_tot_r)))
            hist["train_switch_rate"].append(float(np.mean(ep_sw)))
            if (ep + 1) % eval_every == 0 or ep == 0:
                m = summarise(self.evaluate(eval_eps, seed_base=500_000))
                hist["eval_episode"].append(ep + 1)
                hist["eval_reward"].append(m["reward"])
                hist["eval_rmse"].append(m.get("rmse", float("nan")))
                if verbose:
                    print(f"[{self.fed_mode}] ep {ep+1:5d} train {hist['train_env_reward'][-1]:.3f} "
                          f"eval {m['reward']:.3f} rmse {m.get('rmse', float('nan')):.3f} "
                          f"lat {m['lat']:.0f}ms ({time.time()-t_start:.0f}s)", flush=True)
        self.train_time_s = time.time() - t_start
        hist = dict(hist)
        hist["comm"] = self.fed.meter.as_dict()
        hist["train_time_s"] = self.train_time_s
        return hist
