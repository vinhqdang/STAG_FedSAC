"""
Baselines — SSF, LLF, LSTM-DRL, FedDDPG implementations.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from stag_fedsac.environment.wifi_env import WiFiEnvironment
from stag_fedsac.config import *


class SSFBaseline:
    """Baseline 1: Strongest Signal First — associates users to strongest RSSI AP."""

    def __init__(self, n_aps: int, n_channels: int):
        self.n_aps = n_aps
        self.n_channels = n_channels

    def act(self, obs: dict) -> dict[int, np.ndarray]:
        """Default power/channel, no load balancing."""
        actions = {}
        for i in range(self.n_aps):
            action = np.zeros(1 + self.n_channels + 3)
            action[0] = (P_MIN + P_MAX) / 2  # default power
            action[1 + i % self.n_channels] = 1.0  # fixed channel
            action[1 + self.n_channels:] = [1/3, 1/3, 1/3]  # equal BW
            actions[i] = action
        return actions


class LLFBaseline:
    """Baseline 2: Least Loaded First — reactive load-aware balancing."""

    def __init__(self, n_aps: int, n_channels: int):
        self.n_aps = n_aps
        self.n_channels = n_channels

    def act(self, obs: dict) -> dict[int, np.ndarray]:
        """Adjust power based on load — lower power on overloaded APs."""
        actions = {}
        loads = obs["loads"]

        for i in range(self.n_aps):
            action = np.zeros(1 + self.n_channels + 3)
            # Reduce power if overloaded, increase if underloaded
            if loads[i] > 0.8:
                action[0] = P_MIN + (P_MAX - P_MIN) * 0.3
            elif loads[i] < 0.3:
                action[0] = P_MAX
            else:
                action[0] = P_MIN + (P_MAX - P_MIN) * (1 - loads[i])

            # Round-robin channel assignment
            action[1 + i % self.n_channels] = 1.0

            # BW allocation based on QoS mix
            qos = obs["qos"][i]
            action[1 + self.n_channels:] = np.maximum(qos, 0.1)
            action[1 + self.n_channels:] /= action[1 + self.n_channels:].sum()
            actions[i] = action
        return actions


class LSTMPredictor(nn.Module):
    """LSTM predictor for LSTM-DRL baseline (scalar prediction)."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, T, F] → pred: [batch, 1]"""
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class DDPGActor(nn.Module):
    """DDPG actor for baselines."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class DDPGCritic(nn.Module):
    """DDPG critic for baselines."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))


class LSTMDRLBaseline:
    """Baseline 3: LSTM Predictor + Separate DDPG.

    Scalar LSTM forecast → state augmentation → DDPG (separate training).
    """

    def __init__(self, n_aps: int, n_channels: int, device: torch.device = DEVICE):
        self.n_aps = n_aps
        self.n_channels = n_channels
        self.device = device

        # Scalar LSTM predictor
        self.predictor = LSTMPredictor().to(device)

        # State = local obs (5) + scalar pred (1)
        state_dim = F_FEATURES + 1
        action_dim = 1 + n_channels + 3

        self.actor = DDPGActor(state_dim, action_dim).to(device)
        self.critic = DDPGCritic(state_dim, action_dim).to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.pred_opt = torch.optim.Adam(self.predictor.parameters(), lr=LR_PREDICTOR)

    def act(self, obs: dict, history: list = None) -> dict[int, np.ndarray]:
        actions = {}
        for i in range(self.n_aps):
            h_i = torch.tensor(obs["h"][i], dtype=torch.float32, device=self.device)

            # Scalar prediction
            if history and len(history) >= 2:
                h_seq = torch.tensor(
                    np.stack([h[i] for h in history[-T_HISTORY:]]),
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                with torch.no_grad():
                    pred = self.predictor(h_seq).squeeze()
            else:
                pred = torch.tensor([obs["loads"][i]], device=self.device)

            state = torch.cat([h_i, pred.view(-1)])
            with torch.no_grad():
                action = self.actor(state.unsqueeze(0)).squeeze(0)

            # Scale action to valid range
            action_np = action.cpu().numpy()
            action_out = np.zeros(1 + self.n_channels + 3)
            action_out[0] = (action_np[0] + 1) / 2 * (P_MAX - P_MIN) + P_MIN
            ch_logits = action_np[1:1+self.n_channels]
            action_out[1:1+self.n_channels] = np.exp(ch_logits) / np.exp(ch_logits).sum()
            bw = np.abs(action_np[1+self.n_channels:1+self.n_channels+3]) + 0.01
            action_out[1+self.n_channels:] = bw / bw.sum()
            actions[i] = action_out
        return actions


class FedDDPGBaseline:
    """Baseline 4: Federated DDPG without prediction (adapted from E-FRL)."""

    def __init__(self, n_aps: int, n_channels: int, device: torch.device = DEVICE):
        self.n_aps = n_aps
        self.n_channels = n_channels
        self.device = device

        state_dim = F_FEATURES + 3  # local obs + qos
        action_dim = 1 + n_channels + 3

        self.actors = {
            i: DDPGActor(state_dim, action_dim).to(device)
            for i in range(n_aps)
        }
        self.critics = {
            i: DDPGCritic(state_dim, action_dim).to(device)
            for i in range(n_aps)
        }

        self.actor_opts = {
            i: torch.optim.Adam(self.actors[i].parameters(), lr=LR_ACTOR)
            for i in range(n_aps)
        }

    def act(self, obs: dict) -> dict[int, np.ndarray]:
        actions = {}
        for i in range(self.n_aps):
            h_i = torch.tensor(obs["h"][i], dtype=torch.float32, device=self.device)
            qos_i = torch.tensor(obs["qos"][i], dtype=torch.float32, device=self.device)
            state = torch.cat([h_i, qos_i])

            with torch.no_grad():
                action = self.actors[i](state.unsqueeze(0)).squeeze(0)

            action_np = action.cpu().numpy()
            action_out = np.zeros(1 + self.n_channels + 3)
            action_out[0] = (action_np[0] + 1) / 2 * (P_MAX - P_MIN) + P_MIN
            ch_logits = action_np[1:1+self.n_channels]
            action_out[1:1+self.n_channels] = np.exp(ch_logits - ch_logits.max()) / \
                np.exp(ch_logits - ch_logits.max()).sum()
            bw = np.abs(action_np[1+self.n_channels:1+self.n_channels+3]) + 0.01
            action_out[1+self.n_channels:] = bw / bw.sum()
            actions[i] = action_out
        return actions

    def fedavg(self) -> None:
        """Single-level FedAvg aggregation (no zones)."""
        avg_state = {}
        for name in self.actors[0].state_dict():
            avg_state[name] = torch.mean(
                torch.stack([self.actors[i].state_dict()[name].float() for i in range(self.n_aps)]),
                dim=0,
            )
        for i in range(self.n_aps):
            self.actors[i].load_state_dict(avg_state)


def evaluate_baseline(
    baseline,
    env: WiFiEnvironment,
    n_episodes: int = 20,
    baseline_name: str = "baseline",
) -> dict:
    """Evaluate a baseline over multiple episodes."""
    metrics_all = defaultdict(list)
    history = []

    for ep in range(n_episodes):
        obs = env.reset()
        history = []
        ep_rewards = []
        ep_throughputs = []
        ep_latencies = []
        ep_jain = []
        ep_loads = []

        for step in range(STEPS_PER_EPISODE):
            history.append(obs["h"])

            if hasattr(baseline, "act"):
                if isinstance(baseline, LSTMDRLBaseline):
                    actions = baseline.act(obs, history)
                else:
                    actions = baseline.act(obs)
            else:
                break

            next_obs, rewards, done, info = env.step(actions)

            ep_rewards.append(np.mean(list(rewards.values())))
            ep_throughputs.extend(list(info["throughputs"].values()))
            ep_latencies.extend(list(info["latencies"].values()))
            ep_jain.extend(list(info["jain"].values()))
            ep_loads.extend(list(info["loads"].values()))

            obs = next_obs
            if done:
                break

        metrics_all["reward"].append(np.mean(ep_rewards))
        metrics_all["throughput"].append(np.mean(ep_throughputs))
        metrics_all["latency"].append(np.mean(ep_latencies))
        metrics_all["jain_fairness"].append(np.mean(ep_jain))
        metrics_all["load_std"].append(np.std(ep_loads))

    result = {k: float(np.mean(v)) for k, v in metrics_all.items()}
    result["name"] = baseline_name
    return result
