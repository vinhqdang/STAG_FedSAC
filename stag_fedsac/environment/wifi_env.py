"""
WiFi Environment — Python-based multi-AP WiFi simulation (Gym-compatible).

Simulates a high-density transit venue WiFi network with:
- Multi-AP interference modeling
- QoS-differentiated users (VoIP, Video, BestEffort)
- Schedule-driven crowd surges
- Fair bandwidth allocation tracking
"""
import numpy as np
import torch
from typing import Any


class WiFiEnvironment:
    """Multi-AP WiFi environment for STAG-FedSAC training.

    Each step represents a 5-minute control interval. Actions control
    per-AP power, channel, and bandwidth allocation.
    """

    def __init__(
        self,
        n_aps: int = 30,
        n_users_max: int = 1000,
        n_channels: int = 11,
        p_min: float = 5.0,
        p_max: float = 23.0,
        timestep_s: int = 300,
        episode_len: int = 12,
        adjacency: np.ndarray | None = None,
        zone_assignments: dict[int, int] | None = None,
    ):
        self.n_aps = n_aps
        self.n_users_max = n_users_max
        self.n_channels = n_channels
        self.p_min = p_min
        self.p_max = p_max
        self.timestep_s = timestep_s
        self.episode_len = episode_len
        self.step_count = 0

        # AP positions (2D layout)
        self.ap_positions = self._generate_ap_layout()

        # Interference adjacency matrix
        if adjacency is not None:
            self.adjacency = adjacency
        else:
            self.adjacency = self._build_interference_graph()

        # Zone assignments
        if zone_assignments is not None:
            self.zone_assignments = zone_assignments
        else:
            self.zone_assignments = {
                i: i % 3 for i in range(n_aps)
            }

        # State variables
        self.ap_loads = np.zeros(n_aps)
        self.ap_throughputs = np.zeros(n_aps)
        self.ap_latencies = np.zeros(n_aps)
        self.ap_channel_utils = np.zeros(n_aps)
        self.ap_user_counts = np.zeros(n_aps, dtype=int)
        self.ap_channels = np.zeros(n_aps, dtype=int)
        self.ap_powers = np.full(n_aps, (p_min + p_max) / 2)
        self.ap_bw_alloc = np.ones((n_aps, 3)) / 3  # equal split initially

        # User state
        self.user_positions = np.zeros((0, 2))
        self.user_qos_class = np.zeros(0, dtype=int)  # 0=VoIP, 1=Video, 2=BE
        self.user_ap_assoc = np.zeros(0, dtype=int)

        # QoS parameters
        self.sinr_thresholds = [10.0, 15.0, 5.0]  # VoIP, Video, BE
        self.latency_sla = [50.0, 100.0, 500.0]

        # Schedule state
        self.schedule_events = []
        self.current_schedule_features = np.zeros((n_aps, 8))

    def _generate_ap_layout(self) -> np.ndarray:
        """Generate AP positions in a grid-like indoor layout."""
        n_cols = int(np.ceil(np.sqrt(self.n_aps)))
        n_rows = int(np.ceil(self.n_aps / n_cols))
        positions = []
        spacing = 30.0  # meters
        for i in range(self.n_aps):
            row = i // n_cols
            col = i % n_cols
            x = col * spacing + np.random.randn() * 2
            y = row * spacing + np.random.randn() * 2
            positions.append([x, y])
        return np.array(positions)

    def _build_interference_graph(self) -> np.ndarray:
        """Build interference adjacency based on AP proximity."""
        A = np.zeros((self.n_aps, self.n_aps))
        for i in range(self.n_aps):
            for j in range(i + 1, self.n_aps):
                dist = np.linalg.norm(
                    self.ap_positions[i] - self.ap_positions[j]
                )
                if dist < 80.0:  # interference range
                    # Inverse-distance weight
                    w = max(0, 1.0 - dist / 80.0)
                    A[i, j] = w
                    A[j, i] = w
        return A

    def _generate_users(self, n_users: int) -> None:
        """Generate users with random positions, QoS classes, and AP associations."""
        if n_users <= 0:
            return

        # Random positions within venue bounds
        x_max = self.ap_positions[:, 0].max() + 30
        y_max = self.ap_positions[:, 1].max() + 30
        positions = np.column_stack([
            np.random.uniform(0, x_max, n_users),
            np.random.uniform(0, y_max, n_users),
        ])

        # QoS class distribution: 20% VoIP, 30% Video, 50% BestEffort
        qos = np.random.choice(3, size=n_users, p=[0.2, 0.3, 0.5])

        # Associate to nearest AP
        assoc = np.zeros(n_users, dtype=int)
        for u in range(n_users):
            dists = np.linalg.norm(self.ap_positions - positions[u], axis=1)
            assoc[u] = dists.argmin()

        self.user_positions = positions
        self.user_qos_class = qos
        self.user_ap_assoc = assoc

    def _generate_schedule_events(self) -> list[dict]:
        """Generate synthetic transit schedule events."""
        events = []
        n_events = np.random.randint(3, 8)
        for _ in range(n_events):
            event = {
                "zone": np.random.randint(0, 3),
                "time_offset": np.random.uniform(0, self.episode_len),
                "n_passengers": np.random.randint(50, 300),
                "type": np.random.choice(["departure", "arrival"]),
            }
            events.append(event)
        return events

    def _compute_schedule_features(self, step: int) -> np.ndarray:
        """Compute schedule feature tensor S for all APs at current step."""
        features = np.zeros((self.n_aps, 8))
        sigma = 2.0  # Gaussian kernel width (in timesteps)

        for event in self.schedule_events:
            tau = event["time_offset"] - step
            surge = np.exp(-tau ** 2 / (2 * sigma ** 2))
            zone = event["zone"]

            for ap_id in range(self.n_aps):
                if self.zone_assignments[ap_id] == zone:
                    if event["type"] == "departure":
                        features[ap_id, 0] += event["n_passengers"] * surge
                        features[ap_id, 3] = max(features[ap_id, 3], tau) if tau > 0 else 0
                    else:
                        features[ap_id, 1] += event["n_passengers"] * surge
                        features[ap_id, 4] = max(features[ap_id, 4], -tau) if tau < 0 else 0

                    features[ap_id, 2] = 1.0 if abs(tau) < 3 else 0.0
                    # Zone type one-hot (3 dims)
                    features[ap_id, 5 + zone] = 1.0
                    # Occupancy forecast
                    features[ap_id, 7] = min(1.0, features[ap_id, 0] / 300 + features[ap_id, 1] / 300)

        return features

    def _compute_sinr(self, ap_id: int, power: float, channel: int) -> np.ndarray:
        """Compute per-user SINR for users associated with an AP."""
        user_mask = self.user_ap_assoc == ap_id
        if not user_mask.any():
            return np.array([])

        user_pos = self.user_positions[user_mask]
        # Simple path loss model: L(d) = 38.46 + 20*log10(d) dB
        dist = np.linalg.norm(user_pos - self.ap_positions[ap_id], axis=1)
        dist = np.maximum(dist, 1.0)
        path_loss = 38.46 + 20 * np.log10(dist)

        # Signal
        signal_dbm = power - path_loss

        # Interference from co-channel APs
        interference_mw = 0.0
        for j in range(self.n_aps):
            if j != ap_id and self.ap_channels[j] == channel:
                dist_j = np.linalg.norm(user_pos - self.ap_positions[j], axis=1)
                dist_j = np.maximum(dist_j, 1.0)
                pl_j = 38.46 + 20 * np.log10(dist_j)
                intf_dbm = self.ap_powers[j] - pl_j
                interference_mw += 10 ** (intf_dbm / 10)

        # Noise floor: -95 dBm
        noise_mw = 10 ** (-95.0 / 10)
        signal_mw = 10 ** (signal_dbm / 10)
        sinr = 10 * np.log10(signal_mw / (interference_mw + noise_mw))
        return sinr

    def _compute_throughput(
        self, sinr: np.ndarray, bw_alloc: np.ndarray, n_users_class: np.ndarray
    ) -> float:
        """Shannon-based throughput with per-QoS-class bandwidth allocation.

        Each QoS class receives a fraction of the 20 MHz channel proportional
        to bw_alloc[c]. Users in class c share bw_alloc[c] * 20 MHz equally.
        """
        if len(sinr) == 0:
            return 0.0
        total_bw_mhz = 20.0
        # Normalise allocation (guard against zero-sum)
        bw_alloc = np.asarray(bw_alloc, dtype=float)
        bw_alloc = np.maximum(bw_alloc, 1e-6)
        bw_alloc = bw_alloc / bw_alloc.sum()

        sinr_linear = 10 ** (sinr / 10)
        throughput = 0.0
        for c in range(3):
            class_mask = n_users_class == c
            if not class_mask.any():
                continue
            n_c = class_mask.sum()
            bw_c = bw_alloc[c] * total_bw_mhz  # MHz allocated to class c
            per_user_bw_c = bw_c / n_c          # MHz per user in class c
            throughput += float(
                (per_user_bw_c * np.log2(1 + sinr_linear[class_mask])).sum()
            )
        return throughput

    def reset(self) -> dict[str, Any]:
        """Reset environment for a new episode."""
        self.step_count = 0

        # Generate random user population
        n_users = np.random.randint(
            self.n_users_max // 2, self.n_users_max
        )
        self._generate_users(n_users)

        # Generate schedule events
        self.schedule_events = self._generate_schedule_events()

        # Initialize AP state
        self.ap_channels = np.random.randint(0, self.n_channels, self.n_aps)
        self.ap_powers = np.full(self.n_aps, (self.p_min + self.p_max) / 2)
        self.ap_bw_alloc = np.ones((self.n_aps, 3)) / 3

        # Compute initial metrics
        self._update_ap_metrics()
        self.current_schedule_features = self._compute_schedule_features(0)

        return self._get_obs()

    def _update_ap_metrics(self) -> None:
        """Update per-AP metrics from current state."""
        for i in range(self.n_aps):
            users = self.user_ap_assoc == i
            n_users = users.sum()
            self.ap_user_counts[i] = n_users
            self.ap_loads[i] = min(1.0, n_users / max(self.n_users_max / self.n_aps, 1))

            if n_users > 0:
                sinr = self._compute_sinr(i, self.ap_powers[i], self.ap_channels[i])
                user_classes = self.user_qos_class[users]
                tp = self._compute_throughput(sinr, self.ap_bw_alloc[i], user_classes)
                self.ap_throughputs[i] = tp
                self.ap_latencies[i] = max(5.0, 100.0 / (tp / n_users + 0.1))
                self.ap_channel_utils[i] = min(1.0, n_users / 30)
            else:
                self.ap_throughputs[i] = 0.0
                self.ap_latencies[i] = 0.0
                self.ap_channel_utils[i] = 0.0

    def step(
        self, actions: dict[int, np.ndarray]
    ) -> tuple[dict[str, Any], dict[int, float], bool, dict]:
        """Execute one timestep with per-AP actions.

        Args:
            actions: {ap_id: np.array [power, channel_probs(C), bw(3)]}

        Returns:
            obs, rewards, done, info
        """
        self.step_count += 1

        # Apply actions
        for ap_id, action in actions.items():
            self.ap_powers[ap_id] = np.clip(action[0], self.p_min, self.p_max)
            self.ap_channels[ap_id] = int(action[1:1+self.n_channels].argmax())
            bw = action[1+self.n_channels:1+self.n_channels+3]
            bw = np.maximum(bw, 0.01)
            self.ap_bw_alloc[ap_id] = bw / bw.sum()

        # Simulate user mobility and schedule-driven arrivals
        self._simulate_mobility()
        self._apply_schedule_effects()

        # Update metrics
        self._update_ap_metrics()
        self.current_schedule_features = self._compute_schedule_features(self.step_count)

        # Compute per-AP rewards
        rewards = {}
        info = {"throughputs": {}, "latencies": {}, "jain": {}, "loads": {}}

        for ap_id in range(self.n_aps):
            r, r_info = self._compute_reward(ap_id)
            rewards[ap_id] = r
            info["throughputs"][ap_id] = self.ap_throughputs[ap_id]
            info["latencies"][ap_id] = self.ap_latencies[ap_id]
            info["loads"][ap_id] = self.ap_loads[ap_id]
            info["jain"][ap_id] = r_info["jain"]

        done = self.step_count >= self.episode_len
        return self._get_obs(), rewards, done, info

    def _simulate_mobility(self) -> None:
        """Simple pedestrian mobility model."""
        if len(self.user_positions) == 0:
            return

        # Random walk with small displacement
        displacement = np.random.randn(*self.user_positions.shape) * 2.0  # ~2m per step
        self.user_positions += displacement

        # Re-associate users to nearest AP
        for u in range(len(self.user_positions)):
            dists = np.linalg.norm(
                self.ap_positions - self.user_positions[u], axis=1
            )
            self.user_ap_assoc[u] = dists.argmin()

    def _apply_schedule_effects(self) -> None:
        """Add/remove users based on schedule events (surges)."""
        for event in self.schedule_events:
            tau = abs(event["time_offset"] - self.step_count)
            if tau < 2:  # within 2 timesteps of event
                surge_users = int(event["n_passengers"] * np.exp(-tau ** 2 / 2))
                if event["type"] == "departure" and self.step_count < event["time_offset"]:
                    # Add users heading to gate
                    self._add_surge_users(event["zone"], surge_users)
                elif event["type"] == "arrival":
                    self._add_surge_users(event["zone"], surge_users)

    def _add_surge_users(self, zone: int, n_users: int) -> None:
        """Add surge users near APs in a specific zone."""
        zone_aps = [
            i for i in range(self.n_aps)
            if self.zone_assignments[i] == zone
        ]
        if not zone_aps or n_users <= 0:
            return

        n_users = min(n_users, 100)  # cap per-event surge

        for _ in range(n_users):
            ap = np.random.choice(zone_aps)
            pos = self.ap_positions[ap] + np.random.randn(2) * 10
            self.user_positions = np.vstack([self.user_positions, pos])
            self.user_qos_class = np.append(
                self.user_qos_class, np.random.choice(3, p=[0.2, 0.3, 0.5])
            )
            self.user_ap_assoc = np.append(self.user_ap_assoc, ap)

    def _compute_reward(self, ap_id: int) -> tuple[float, dict]:
        """Compute reward for an AP following the algorithm design."""
        users = self.user_ap_assoc == ap_id
        n_users = users.sum()

        if n_users == 0:
            return 0.0, {"jain": 1.0, "qos_violations": 0}

        # Throughput reward (normalized)
        tp_max = 100.0  # max throughput Mbps baseline
        r_tp = min(1.0, self.ap_throughputs[ap_id] / tp_max)

        # Latency penalty (normalized)
        lat_sla = 100.0  # ms
        r_lat = -min(1.0, self.ap_latencies[ap_id] / lat_sla)

        # Jain's fairness
        sinr = self._compute_sinr(ap_id, self.ap_powers[ap_id], self.ap_channels[ap_id])
        if len(sinr) > 1:
            sinr_pos = np.maximum(sinr, 0.1)
            jain = (sinr_pos.sum() ** 2) / (len(sinr_pos) * (sinr_pos ** 2).sum())
        else:
            jain = 1.0

        # QoS violation count
        user_classes = self.user_qos_class[users]
        qos_violations = 0
        for idx, s in enumerate(sinr):
            if idx < len(user_classes) and s < self.sinr_thresholds[user_classes[idx]]:
                qos_violations += 1
        r_qos = -qos_violations / max(n_users, 1)

        # NOTE: capacity constraint is handled by the adaptive Lagrangian
        # multiplier in JointTrainer.  No r_cap term here to avoid
        # double-counting (design spec section 6.2, constraint C2).

        # Combined reward
        reward = (
            0.40 * r_tp
            + 0.25 * r_lat
            + 0.20 * jain
            + 0.05 * r_qos
        )

        return float(reward), {
            "jain": float(jain),
            "qos_violations": qos_violations,
            "throughput": float(self.ap_throughputs[ap_id]),
            "latency": float(self.ap_latencies[ap_id]),
        }

    def _get_obs(self) -> dict[str, Any]:
        """Get current observation dict."""
        # Per-AP local observations [F=5]
        h = np.column_stack([
            self.ap_loads,
            self.ap_throughputs / 100.0,  # normalize
            self.ap_latencies / 500.0,
            self.ap_channel_utils,
            self.ap_user_counts / max(self.n_users_max / self.n_aps, 1),
        ])

        # QoS fractions per AP [3]
        qos_fracs = np.zeros((self.n_aps, 3))
        for i in range(self.n_aps):
            users = self.user_ap_assoc == i
            n = users.sum()
            if n > 0:
                classes = self.user_qos_class[users]
                for c in range(3):
                    qos_fracs[i, c] = (classes == c).sum() / n

        return {
            "h": h.astype(np.float32),                                    # [N, F]
            "qos": qos_fracs.astype(np.float32),                         # [N, 3]
            "schedule": self.current_schedule_features.astype(np.float32),# [N, D_s]
            "adjacency": self.adjacency.astype(np.float32),              # [N, N]
            "loads": self.ap_loads.copy().astype(np.float32),            # [N]
            "true_loads": self.ap_loads.copy().astype(np.float32),       # [N]
        }

    def get_adjacency_tensor(self) -> torch.Tensor:
        """Return adjacency matrix as tensor."""
        return torch.tensor(self.adjacency, dtype=torch.float32)
