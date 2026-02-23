"""
Joint Trainer — Main training loop implementing Algorithm steps 1–16.
Core novelty: bidirectional gradient flow between predictor and policy.
"""
import os
import copy
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from stag_fedsac.models.stgcat import STGCAT
from stag_fedsac.models.graph_sac import PersonalizedSACActor, SACCriticNetwork
from stag_fedsac.models.hierfed_kd import HierFedKD
from stag_fedsac.training.replay_buffer import ReplayBuffer
from stag_fedsac.training.lagrangian import LagrangianManager
from stag_fedsac.environment.wifi_env import WiFiEnvironment
from stag_fedsac.environment.schedule_generator import ScheduleGenerator
from stag_fedsac.config import *


class JointTrainer:
    """STAG-FedSAC joint training with bidirectional gradient flow.

    Implements the full Algorithm (steps 1–16) from the design document.
    """

    def __init__(
        self,
        n_aps: int = N_AP,
        n_channels: int = N_CHANNELS,
        device: torch.device = DEVICE,
        log_dir: str = LOG_DIR,
        checkpoint_dir: str = CHECKPOINT_DIR,
        use_joint_training: bool = True,
        use_schedule: bool = True,
        use_hierarchical_fed: bool = True,
    ):
        self.n_aps = n_aps
        self.n_channels = n_channels
        self.device = device
        self.use_joint_training = use_joint_training
        self.use_schedule = use_schedule
        self.use_hierarchical_fed = use_hierarchical_fed

        # Zone assignments
        self.zone_assignments = {i: i % N_ZONES for i in range(n_aps)}

        # ── Initialize modules ──
        # Module 1: ST-GCAT Predictor
        self.stgcat = STGCAT(
            n_features=F_FEATURES,
            d_schedule=D_SCHEDULE,
            d_hidden=D_HIDDEN,
            n_heads=N_HEADS,
            t_history=T_HISTORY,
            delta=DELTA_HORIZON,
            n_transformer_layers=N_TRANSFORMER_LAYERS,
            dropout=DROPOUT,
        ).to(device)

        # Module 2: Per-AP SAC Agents
        state_dim = F_FEATURES + DELTA_HORIZON * D_HIDDEN + 3 + D_SCHEDULE
        action_dim = 1 + n_channels + 3  # power + channel + bw

        self.actors: dict[int, PersonalizedSACActor] = {}
        self.critics: dict[int, SACCriticNetwork] = {}
        self.target_critics: dict[int, SACCriticNetwork] = {}
        self.replay_buffers: dict[int, ReplayBuffer] = {}

        for i in range(n_aps):
            self.actors[i] = PersonalizedSACActor(
                state_dim=state_dim,
                n_channels=n_channels,
                p_min=P_MIN,
                p_max=P_MAX,
            ).to(device)

            self.critics[i] = SACCriticNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
            ).to(device)

            self.target_critics[i] = copy.deepcopy(self.critics[i])
            self.target_critics[i].requires_grad_(False)

            self.replay_buffers[i] = ReplayBuffer(
                REPLAY_BUFFER_SIZE, state_dim, action_dim, device
            )

        # Entropy temperature (auto-tuned)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.target_entropy = -action_dim

        # Module 3: HierFed-KD
        self.hierfed = HierFedKD(
            zone_assignments=self.zone_assignments,
            n_zones=N_ZONES,
            fedprox_mu=FEDPROX_MU,
            lr_kd=LR_FEDERATED_KD,
            s_ref_size=S_REF_SIZE,
            device=device,
        )

        # Lagrangian manager
        self.lagrangian = LagrangianManager(n_aps, L_MAX, RHO_LAGRANGIAN, device)

        # Optimizers
        self.pred_optimizer = torch.optim.Adam(
            self.stgcat.parameters(), lr=LR_PREDICTOR
        )
        self.actor_optimizers = {
            i: torch.optim.Adam(self.actors[i].parameters(), lr=LR_ACTOR)
            for i in range(n_aps)
        }
        self.critic_optimizers = {
            i: torch.optim.Adam(self.critics[i].parameters(), lr=LR_CRITIC)
            for i in range(n_aps)
        }
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=LR_ACTOR)

        # Environment
        self.env = WiFiEnvironment(
            n_aps=n_aps,
            n_users_max=N_USERS_MAX,
            n_channels=n_channels,
            p_min=P_MIN,
            p_max=P_MAX,
            timestep_s=TIMESTEP_S,
            episode_len=STEPS_PER_EPISODE,
            zone_assignments=self.zone_assignments,
        )

        self.schedule_gen = ScheduleGenerator(
            n_aps=n_aps,
            delta=DELTA_HORIZON,
            d_schedule=D_SCHEDULE,
            zone_assignments=self.zone_assignments,
        )

        # Logging
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.checkpoint_dir = checkpoint_dir

        # History tracking
        self.history_buffer = {}  # per-AP history of observations
        self.global_step = 0

    def _build_state(
        self,
        obs: dict,
        z_fused: torch.Tensor,
        ap_id: int,
    ) -> torch.Tensor:
        """Construct SAC state: concat[h_i, Z_fused_i.flatten(), qos_i, sched_i]."""
        h_i = torch.tensor(obs["h"][ap_id], dtype=torch.float32, device=self.device)
        z_i = z_fused[0, ap_id].flatten()  # [delta * d_h]
        qos_i = torch.tensor(obs["qos"][ap_id], dtype=torch.float32, device=self.device)
        sched_i = torch.tensor(
            obs["schedule"][ap_id], dtype=torch.float32, device=self.device
        )
        return torch.cat([h_i, z_i, qos_i, sched_i])

    def _get_history_tensor(self, obs: dict) -> torch.Tensor:
        """Build historical observation tensor H from buffer.

        Returns: [1, N, T, F] tensor
        """
        h = obs["h"]  # [N, F]

        if "history" not in self.history_buffer:
            self.history_buffer["history"] = []

        self.history_buffer["history"].append(h)
        if len(self.history_buffer["history"]) > T_HISTORY:
            self.history_buffer["history"].pop(0)

        # Pad if not enough history
        history = list(self.history_buffer["history"])
        while len(history) < T_HISTORY:
            history.insert(0, history[0])

        H = np.stack(history, axis=1)  # [N, T, F]
        return torch.tensor(H, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _get_schedule_tensor(self, obs: dict) -> torch.Tensor:
        """Build schedule feature tensor S.

        Returns: [1, N, delta, D_s]
        """
        S = self.schedule_gen.generate_schedule_tensor(
            self.env.step_count,
            self.env.schedule_events,
        )
        return torch.tensor(S, dtype=torch.float32, device=self.device).unsqueeze(0)

    def train(
        self,
        total_episodes: int = TOTAL_EPISODES,
        eval_interval: int = 50,
    ) -> dict:
        """Main training loop — Algorithm steps 1–16."""
        print(f"\n{'='*60}")
        print(f"STAG-FedSAC Joint Training")
        print(f"  Joint training: {self.use_joint_training}")
        print(f"  Schedule features: {self.use_schedule}")
        print(f"  Hierarchical FL: {self.use_hierarchical_fed}")
        print(f"  Device: {self.device}")
        print(f"  N_APs: {self.n_aps}")
        print(f"{'='*60}\n")

        best_reward = -float("inf")
        training_history = defaultdict(list)

        for episode in range(total_episodes):
            ep_start = time.time()
            obs = self.env.reset()
            self.history_buffer = {}
            episode_rewards = defaultdict(float)
            episode_info = defaultdict(list)

            for step in range(STEPS_PER_EPISODE):
                self.global_step += 1

                # ── [1-2] OBSERVE & PREDICT ──
                A = self.env.get_adjacency_tensor().to(self.device)
                H = self._get_history_tensor(obs)  # [1, N, T, F]
                S = self._get_schedule_tensor(obs) if self.use_schedule else \
                    torch.zeros(1, self.n_aps, DELTA_HORIZON, D_SCHEDULE, device=self.device)

                with torch.no_grad():
                    Z_fused, L_hat = self.stgcat(H, S, A)

                # ── [3-4] CONSTRUCT STATE & SAMPLE ACTION ──
                actions = {}
                states = {}
                log_probs = {}

                for i in range(self.n_aps):
                    s_i = self._build_state(obs, Z_fused, i)
                    states[i] = s_i

                    if self.global_step < WARMUP_STEPS:
                        # Random action during warmup
                        action = torch.randn(1 + self.n_channels + 3, device=self.device)
                        action[0] = torch.rand(1, device=self.device) * (P_MAX - P_MIN) + P_MIN
                        action[1:1+self.n_channels] = F.softmax(action[1:1+self.n_channels], dim=-1)
                        action[1+self.n_channels:] = F.softmax(action[1+self.n_channels:], dim=-1)
                        log_probs[i] = torch.tensor(0.0, device=self.device)
                    else:
                        action, log_prob, _, _ = self.actors[i](
                            s_i.unsqueeze(0)
                        )
                        action = action.squeeze(0)
                        log_probs[i] = log_prob.squeeze(0)

                    actions[i] = action.detach().cpu().numpy()

                # ── [5] ENVIRONMENT STEP ──
                next_obs, rewards, done, info = self.env.step(actions)

                # ── [6] COMPUTE REWARD (with prediction bonus) ──
                for i in range(self.n_aps):
                    # Prediction bonus: -|L_hat - L_true|
                    pred_bonus = -abs(
                        L_hat[0, i, 0].item() - obs["true_loads"][i]
                    )
                    # Lagrangian penalty
                    lag_penalty = self.lagrangian.get_penalty(
                        i, self.env.ap_loads[i]
                    )

                    total_reward = rewards[i] + DELTA_REWARD * pred_bonus + lag_penalty
                    episode_rewards[i] += total_reward

                    # ── [7] STORE TRANSITION ──
                    next_A = self.env.get_adjacency_tensor().to(self.device)
                    next_H = self._get_history_tensor(next_obs)
                    next_S = self._get_schedule_tensor(next_obs) if self.use_schedule else \
                        torch.zeros(1, self.n_aps, DELTA_HORIZON, D_SCHEDULE, device=self.device)

                    with torch.no_grad():
                        next_Z_fused, _ = self.stgcat(next_H, next_S, next_A)

                    next_s_i = self._build_state(next_obs, next_Z_fused, i)

                    self.replay_buffers[i].store(
                        states[i],
                        torch.tensor(actions[i], dtype=torch.float32),
                        total_reward,
                        next_s_i,
                        done,
                    )

                    # Update reward tracking for federation
                    self.hierfed.update_reward(i, total_reward)

                # ── [8-14] BACKWARD PASS ──
                if (
                    self.global_step >= WARMUP_STEPS
                    and self.global_step % UPDATE_FREQ == 0
                ):
                    self._update_step(obs, A)

                # ── [13] LAGRANGIAN UPDATE ──
                loads_tensor = torch.tensor(
                    self.env.ap_loads, dtype=torch.float32, device=self.device
                )
                self.lagrangian.update(loads_tensor)

                # ── [15-16] FEDERATED SYNCHRONIZATION ──
                if self.use_hierarchical_fed:
                    if self.global_step % T_LOCAL_FED == 0:
                        self.hierfed.zone_aggregation(self.actors)
                        self.hierfed.update_reference_states(
                            self.replay_buffers,
                            F_FEATURES + DELTA_HORIZON * D_HIDDEN + 3 + D_SCHEDULE,
                        )

                    if self.global_step % T_GLOBAL_FED == 0:
                        self.hierfed.global_aggregation(self.actors)

                obs = next_obs

                if done:
                    break

            # ── Episode logging ──
            ep_time = time.time() - ep_start
            avg_reward = np.mean(list(episode_rewards.values()))
            training_history["episode_reward"].append(avg_reward)
            training_history["episode_time"].append(ep_time)

            self.writer.add_scalar("train/avg_reward", avg_reward, episode)
            self.writer.add_scalar("train/episode_time", ep_time, episode)

            if episode % 10 == 0:
                print(
                    f"Episode {episode:4d} | "
                    f"Avg Reward: {avg_reward:.4f} | "
                    f"Time: {ep_time:.1f}s | "
                    f"Steps: {self.global_step}"
                )

            # ── Evaluation ──
            if episode % eval_interval == 0 and episode > 0:
                eval_metrics = self.evaluate()
                for k, v in eval_metrics.items():
                    self.writer.add_scalar(f"eval/{k}", v, episode)
                    training_history[f"eval_{k}"].append(v)

                print(f"  Eval | {eval_metrics}")

                if avg_reward > best_reward:
                    best_reward = avg_reward
                    self.save_checkpoint("best")

            # Periodic checkpoint
            if episode % 100 == 0:
                self.save_checkpoint(f"ep{episode}")

        self.writer.close()
        return dict(training_history)

    def _update_step(self, obs: dict, A: torch.Tensor) -> None:
        """Execute backward pass — Algorithm steps 8–14."""
        alpha = torch.exp(self.log_alpha).detach()

        for i in range(self.n_aps):
            if len(self.replay_buffers[i]) < BATCH_SIZE:
                continue

            # ── [8] SAMPLE BATCH ──
            s, a, r, s_next, done = self.replay_buffers[i].sample(BATCH_SIZE)

            # ── [9] CRITIC UPDATE ──
            with torch.no_grad():
                a_next, log_pi_next, _, _ = self.actors[i](s_next)
                q1_target, q2_target = self.target_critics[i](s_next, a_next)
                q_target = torch.min(q1_target, q2_target) - alpha * log_pi_next
                y = r + GAMMA_DISCOUNT * (1 - done) * q_target

            q1, q2 = self.critics[i](s, a)
            critic_loss = 0.5 * F.mse_loss(q1, y) + 0.5 * F.mse_loss(q2, y)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), 1.0)
            self.critic_optimizers[i].step()

            # ── [10] ACTOR UPDATE ──
            a_curr, log_pi_curr, _, _ = self.actors[i](s)
            q1_curr, q2_curr = self.critics[i](s, a_curr)
            q_min = torch.min(q1_curr, q2_curr)
            actor_loss = (alpha * log_pi_curr - q_min).mean()

            # Add FedProx term
            if self.use_hierarchical_fed:
                prox_loss = self.hierfed.compute_fedprox_loss(self.actors[i])
                actor_loss = actor_loss + prox_loss

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 1.0)
            self.actor_optimizers[i].step()

            # ── [11] ENTROPY TEMPERATURE UPDATE ──
            alpha_loss = -(
                self.log_alpha * (log_pi_curr.detach() + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # ── [14] TARGET NETWORK SOFT UPDATE ──
            with torch.no_grad():
                for tp, p in zip(
                    self.target_critics[i].parameters(),
                    self.critics[i].parameters(),
                ):
                    tp.data.copy_(TAU_SOFT * p.data + (1 - TAU_SOFT) * tp.data)

        # ── [12] JOINT GRADIENT (CORE NOVELTY) ──
        if self.use_joint_training:
            self._joint_gradient_step(obs, A)

    def _joint_gradient_step(self, obs: dict, A: torch.Tensor) -> None:
        """Step 12: Joint gradient — policy gradient flows into predictor.

        L_pred = MSE(L_hat, L_true) + η · L_actor_fresh
        """
        alpha = torch.exp(self.log_alpha).detach()

        # Get fresh prediction with gradient
        H = self._get_history_tensor(obs)
        S = self._get_schedule_tensor(obs) if self.use_schedule else \
            torch.zeros(1, self.n_aps, DELTA_HORIZON, D_SCHEDULE, device=self.device)
        Z_fused_fresh, L_hat_fresh = self.stgcat(H, S, A)

        # Standard prediction loss
        L_true = torch.tensor(
            obs["true_loads"], dtype=torch.float32, device=self.device
        )
        pred_loss = F.mse_loss(L_hat_fresh[0, :, 0], L_true)

        # Policy-informed gradient: run actors with fresh embeddings
        actor_loss_total = torch.tensor(0.0, device=self.device)
        n_valid = 0

        for i in range(self.n_aps):
            if len(self.replay_buffers[i]) < BATCH_SIZE:
                continue

            s_fresh = self._build_state(obs, Z_fused_fresh, i)
            a_fresh, log_pi_fresh, _, _ = self.actors[i](s_fresh.unsqueeze(0))

            q1_fresh, q2_fresh = self.critics[i](
                s_fresh.unsqueeze(0), a_fresh
            )
            q_min_fresh = torch.min(q1_fresh, q2_fresh)
            actor_loss_i = (alpha * log_pi_fresh - q_min_fresh).mean()
            actor_loss_total = actor_loss_total + actor_loss_i
            n_valid += 1

        if n_valid > 0:
            actor_loss_total = actor_loss_total / n_valid

        # Joint loss (bidirectional gradient bridge)
        joint_loss = pred_loss + ETA_JOINT * actor_loss_total

        self.pred_optimizer.zero_grad()
        joint_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.stgcat.parameters(), 1.0)
        self.pred_optimizer.step()

    def evaluate(self, n_episodes: int = 5) -> dict:
        """Evaluate current policy over test episodes."""
        self.stgcat.eval()
        for actor in self.actors.values():
            actor.eval()

        metrics = defaultdict(list)

        for _ in range(n_episodes):
            obs = self.env.reset()
            self.history_buffer = {}
            ep_rewards = []
            ep_throughputs = []
            ep_latencies = []
            ep_jain = []
            ep_loads = []

            for step in range(STEPS_PER_EPISODE):
                A = self.env.get_adjacency_tensor().to(self.device)
                H = self._get_history_tensor(obs)
                S = self._get_schedule_tensor(obs) if self.use_schedule else \
                    torch.zeros(1, self.n_aps, DELTA_HORIZON, D_SCHEDULE, device=self.device)

                with torch.no_grad():
                    Z_fused, L_hat = self.stgcat(H, S, A)

                actions = {}
                for i in range(self.n_aps):
                    s_i = self._build_state(obs, Z_fused, i)
                    action, _, _, _ = self.actors[i](
                        s_i.unsqueeze(0), deterministic=True
                    )
                    actions[i] = action.squeeze(0).cpu().numpy()

                next_obs, rewards, done, info = self.env.step(actions)

                ep_rewards.append(np.mean(list(rewards.values())))
                ep_throughputs.extend(list(info["throughputs"].values()))
                ep_latencies.extend(list(info["latencies"].values()))
                ep_jain.extend(list(info["jain"].values()))
                ep_loads.extend(list(info["loads"].values()))

                # Prediction error
                pred_error = np.mean(np.abs(
                    L_hat[0, :, 0].cpu().numpy() - obs["true_loads"]
                ))
                metrics["pred_mae"].append(pred_error)

                obs = next_obs
                if done:
                    break

            metrics["reward"].append(np.mean(ep_rewards))
            metrics["throughput"].append(np.mean(ep_throughputs))
            metrics["latency"].append(np.mean(ep_latencies))
            metrics["jain_fairness"].append(np.mean(ep_jain))
            metrics["load_std"].append(np.std(ep_loads))

        self.stgcat.train()
        for actor in self.actors.values():
            actor.train()

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def save_checkpoint(self, tag: str) -> None:
        """Save all model states."""
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{tag}.pt")
        state = {
            "stgcat": self.stgcat.state_dict(),
            "actors": {i: a.state_dict() for i, a in self.actors.items()},
            "critics": {i: c.state_dict() for i, c in self.critics.items()},
            "log_alpha": self.log_alpha.detach().cpu(),
            "global_step": self.global_step,
        }
        torch.save(state, path)

    def load_checkpoint(self, tag: str) -> None:
        """Load checkpoint."""
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{tag}.pt")
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.stgcat.load_state_dict(state["stgcat"])
        for i, sd in state["actors"].items():
            self.actors[i].load_state_dict(sd)
        for i, sd in state["critics"].items():
            self.critics[i].load_state_dict(sd)
        self.global_step = state["global_step"]
