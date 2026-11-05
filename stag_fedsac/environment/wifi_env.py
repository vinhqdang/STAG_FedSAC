"""
WiFi Environment -- multi-AP transit-venue simulator.

Physical / MAC model (quasi-static within one 5-minute control interval):

* Path loss  PL(d) = 46.4 + 30 log10(d) dB (5 GHz, indoor, exponent 3.0).
* C orthogonal 20 MHz channels (no adjacent-channel leakage).
* APs that can hear each other (RSSI >= CS_THRESHOLD_DBM) and share a channel
  form a carrier-sense domain and TIME-SHARE the medium (airtime proportional to
  associated users); only co-channel APs outside the domain add interference.
* MAC contention: aggregate efficiency eta(m) from Bianchi's 802.11 DCF model
  for m active stations in the domain (``mac_model``), and a mean access delay
  derived from the same model.
* PHY rate: 0.75 * B * log2(1 + SINR), capped at the 2-stream VHT maximum.
* QoS classes (VoIP / video / best-effort) map to 802.11 access categories
  (AC_VO / AC_VI / AC_BE). The bandwidth action is an airtime budget per AC,
  quantised to 1/16 units; unused budget is redistributed (work conserving).
* Channel switching: a channel change costs ``switch_outage_s`` seconds of
  service (channel-switch announcement + client re-association).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from stag_fedsac.environment.mac_model import EfficiencyTable, quantise_simplex, quantise_simplex_batch

QOS_NAMES = ("VoIP", "Video", "BestEffort")


class WiFiEnvironment:
    """Multi-AP WiFi environment. One step = one 5-minute control interval."""

    # ---- physical constants -------------------------------------------------
    PL_REF_DB = 46.4
    PL_EXP = 3.0
    NOISE_DBM = -95.0
    CS_THRESHOLD_DBM = -82.0
    BANDWIDTH_MHZ = 20.0
    PHY_MAX_MBPS = 156.0          # 2 spatial streams, 256-QAM 5/6, 20 MHz
    SHANNON_GAP = 0.75
    LOAD_REF_USERS = 60.0         # users/AP that map to a normalised load of 1
    ACTIVE_FRACTION = 0.25        # fraction of associated users with backlogged traffic
    BASE_LATENCY_MS = 5.0         # wired backhaul + stack latency
    MAX_LATENCY_MS = 500.0        # drop-tail buffer limit
    USER_LIFETIME_SURGE = 2       # steps a surge user stays in the venue
    AP_STATIC_POWER_W = 6.0       # baseband + radio front-end draw of an AP
    PA_EFFICIENCY = 0.3           # power-amplifier efficiency (radiated / DC power)

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
        seed: int = 0,
        switch_outage_s: float = 3.0,
        mac_contention: bool = True,
        edca_levels: int = 16,
        reward_weights: dict | None = None,
        util_limit: float = 0.9,
    ):
        self.n_aps = n_aps
        self.n_users_max = n_users_max
        self.n_channels = n_channels
        self.p_min = p_min
        self.p_max = p_max
        self.timestep_s = timestep_s
        self.episode_len = episode_len
        self.switch_outage_s = switch_outage_s
        self.mac_contention = mac_contention
        self.edca_levels = edca_levels
        self.util_limit = util_limit
        self.step_count = 0

        w = dict(tp=0.40, lat=0.25, fair=0.20, qos=0.05)
        if reward_weights:
            w.update(reward_weights)
        self.rw = w

        self.rng = np.random.default_rng(seed)
        self.layout_seed = seed

        # QoS classes: VoIP, Video, Best-effort
        self.sinr_thresholds = np.array([10.0, 15.0, 5.0])
        self.latency_sla = np.array([50.0, 100.0, 500.0])
        self.demand_mbps = np.array([0.1, 3.0, 1.0])       # mean offered rate per user
        self.class_probs = np.array([0.2, 0.3, 0.5])

        self.eff_table = EfficiencyTable(n_max=400, phy_rate_mbps=self.PHY_MAX_MBPS)

        self.ap_positions = self._generate_ap_layout()
        self.adjacency = adjacency if adjacency is not None else self._build_interference_graph()
        self.zone_assignments = zone_assignments if zone_assignments is not None else {
            i: i % 3 for i in range(n_aps)
        }
        self.zone_vec = np.array([self.zone_assignments[i] for i in range(n_aps)])

        # AP-AP distances (static)
        d = np.linalg.norm(self.ap_positions[:, None, :] - self.ap_positions[None, :, :], axis=-1)
        self._ap_dist = np.maximum(d, 1.0)

        # State variables
        self.ap_loads = np.zeros(n_aps)          # normalised user load (exogenous, predicted)
        self.ap_util = np.zeros(n_aps)           # airtime utilisation (controllable, constrained)
        self.ap_throughputs = np.zeros(n_aps)
        self.ap_latencies = np.zeros(n_aps)
        self.ap_channel_utils = np.zeros(n_aps)
        self.ap_user_counts = np.zeros(n_aps, dtype=int)
        self.ap_channels = np.zeros(n_aps, dtype=int)
        self.ap_powers = np.full(n_aps, (p_min + p_max) / 2)
        self.ap_bw_alloc = np.ones((n_aps, 3)) / 3
        self.switched = np.zeros(n_aps, dtype=bool)
        self.ap_domain_users = np.zeros(n_aps)
        self.ap_domain_aps = np.zeros(n_aps)

        self.user_positions = np.zeros((0, 2))
        self.user_qos_class = np.zeros(0, dtype=int)
        self.user_ap_assoc = np.zeros(0, dtype=int)
        self.user_expire = np.zeros(0)

        self.schedule_events: list[dict] = []
        self.last_sys: dict = {}
        self._user_sat = np.zeros(0)

    # ------------------------------------------------------------------ layout
    def _generate_ap_layout(self) -> np.ndarray:
        n_cols = int(np.ceil(np.sqrt(self.n_aps)))
        spacing = 30.0
        pos = []
        for i in range(self.n_aps):
            row, col = divmod(i, n_cols)
            pos.append([col * spacing + self.rng.normal() * 2, row * spacing + self.rng.normal() * 2])
        return np.array(pos)

    def _build_interference_graph(self) -> np.ndarray:
        """Static interference graph: weight decays linearly with AP distance."""
        d = np.linalg.norm(self.ap_positions[:, None] - self.ap_positions[None], axis=-1)
        A = np.where((d < 80.0) & (d > 0), 1.0 - d / 80.0, 0.0)
        return np.clip(A, 0, None)

    # ------------------------------------------------------------------- users
    def _pathloss(self, d: np.ndarray) -> np.ndarray:
        return self.PL_REF_DB + 10 * self.PL_EXP * np.log10(np.maximum(d, 1.0))

    def _nearest_ap(self, pos: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(pos[:, None, :] - self.ap_positions[None, :, :], axis=-1)
        return d.argmin(axis=1)

    def _generate_users(self, n_users: int) -> None:
        x_max = self.ap_positions[:, 0].max() + 30
        y_max = self.ap_positions[:, 1].max() + 30
        pos = np.column_stack([self.rng.uniform(0, x_max, n_users), self.rng.uniform(0, y_max, n_users)])
        self.user_positions = pos
        self.user_qos_class = self.rng.choice(3, size=n_users, p=self.class_probs)
        self.user_ap_assoc = self._nearest_ap(pos)
        self.user_expire = np.full(n_users, np.inf)

    def _generate_schedule_events(self) -> list[dict]:
        events = []
        for _ in range(int(self.rng.integers(3, 8))):
            events.append({
                "zone": int(self.rng.integers(0, 3)),
                "time_offset": float(self.rng.uniform(0, self.episode_len)),
                "n_passengers": int(self.rng.integers(50, 300)),
                "type": str(self.rng.choice(["departure", "arrival"])),
            })
        return events

    # ----------------------------------------------------------------- physics
    def _ap_domain_matrix(self) -> np.ndarray:
        """cs[i, j] = True if APs i and j can hear each other (symmetric)."""
        rssi = self.ap_powers[None, :] - self._pathloss(self._ap_dist)        # rssi[i, j]: j heard at i
        np.fill_diagonal(rssi, -np.inf)
        heard = np.maximum(rssi, rssi.T) >= self.CS_THRESHOLD_DBM
        return heard

    def _compute_network(self) -> dict:
        """Evaluate one control interval; returns per-AP and per-user quantities."""
        n_ap = self.n_aps
        U = len(self.user_positions)
        assoc = self.user_ap_assoc
        cls = self.user_qos_class
        counts = np.bincount(assoc, minlength=n_ap).astype(float)
        self.ap_user_counts = counts.astype(int)
        self.ap_loads = np.minimum(1.0, counts / self.LOAD_REF_USERS)

        if U == 0:
            self.ap_util[:] = 0
            self.ap_throughputs[:] = 0
            self.ap_latencies[:] = 0
            self.ap_channel_utils[:] = 0
            self._user_sat = np.zeros(0)
            return {"user_tput": np.zeros(0), "user_lat": np.zeros(0), "user_sinr": np.zeros(0),
                    "qos_ok": np.zeros(0, dtype=bool), "ap_jain": np.ones(n_ap)}

        same_ch = self.ap_channels[:, None] == self.ap_channels[None, :]
        cs = self._ap_domain_matrix()
        if self.mac_contention:
            domain = (same_ch & cs) | np.eye(n_ap, dtype=bool)
        else:
            domain = np.eye(n_ap, dtype=bool)
        interferers = same_ch & ~domain                     # co-channel APs outside the CS domain

        # --- SINR per user -----------------------------------------------------
        d_u_ap = np.linalg.norm(self.user_positions[:, None, :] - self.ap_positions[None, :, :], axis=-1)
        rx_dbm = self.ap_powers[None, :] - self._pathloss(d_u_ap)             # [U, N]
        rx_mw = 10 ** (rx_dbm / 10)
        sig_mw = rx_mw[np.arange(U), assoc]
        intf_mask = interferers[assoc]                                        # [U, N]
        intf_mw = (rx_mw * intf_mask).sum(axis=1)
        noise_mw = 10 ** (self.NOISE_DBM / 10)
        sinr_db = 10 * np.log10(sig_mw / (intf_mw + noise_mw))
        sinr_lin = 10 ** (sinr_db / 10)
        phy = np.minimum(self.PHY_MAX_MBPS,
                         self.SHANNON_GAP * self.BANDWIDTH_MHZ * np.log2(1 + sinr_lin))
        phy = np.maximum(phy, 1.0)                                            # lowest MCS floor

        # --- contention domain: airtime share and MAC efficiency ----------------
        dom_users = domain @ counts                                           # users in the domain
        self.ap_domain_users = dom_users
        self.ap_domain_aps = domain.sum(axis=1) - 1                           # co-channel APs in the domain
        air_share = np.where(counts > 0, counts / np.maximum(dom_users, 1.0), 0.0)
        m_active = np.maximum(1, np.rint(dom_users * self.ACTIVE_FRACTION)).astype(int)
        eta = self.eff_table.efficiency(m_active) if self.mac_contention else np.ones(n_ap)
        svc_ms = self.eff_table.service_ms(m_active) if self.mac_contention else np.full(n_ap, 1.5)
        avail = air_share * eta                                               # payload airtime fraction

        # channel-switch outage: lost service fraction and delayed packets
        f_out = np.where(self.switched, min(1.0, self.switch_outage_s / self.timestep_s), 0.0)

        demand_u = self.demand_mbps[cls]
        need_u = demand_u / phy                                               # airtime needed per user

        # airtime needed per AP and class, then work-conserving allocation (vectorised over APs)
        need_c = np.bincount(assoc * 3 + cls, weights=need_u, minlength=n_ap * 3).reshape(n_ap, 3)
        present = need_c > 0
        ok = (counts > 0) & (avail > 1e-9)
        avail_s = np.where(ok, avail, 1.0)
        util = np.where(ok, need_c.sum(axis=1) / avail_s, 0.0)
        share = np.where(present, self.ap_bw_alloc, 0.0)
        share = share / np.maximum(share.sum(axis=1, keepdims=True), 1e-9)
        granted = np.minimum(share * avail_s[:, None], need_c)
        leftover = avail_s - granted.sum(axis=1)
        deficit = need_c - granted
        dsum = deficit.sum(axis=1)
        redistribute = (leftover > 1e-12) & (dsum > 1e-12)
        extra = np.minimum(deficit, leftover[:, None] * deficit / np.maximum(dsum, 1e-12)[:, None])
        granted = granted + np.where(redistribute[:, None], extra, 0.0)
        frac_c = np.where(present, granted / np.maximum(need_c, 1e-12), 0.0)
        rho_c = np.where(granted > 0, need_c / np.maximum(granted, 1e-12), 10.0)
        rho_c = np.where(present, rho_c, 0.0)
        # delay: M/G/1-type queueing on the domain service time (Bianchi), per-AC utilisation
        rho = np.minimum(rho_c, 0.98)
        soj = svc_ms[:, None] * (1.0 + rho / (2.0 * (1.0 - rho)))
        lat_c = self.BASE_LATENCY_MS + soj
        over = np.clip((rho_c - 0.98) / 0.5, 0.0, 1.0)                        # queue fills towards the buffer limit
        lat_c = lat_c + over * (self.MAX_LATENCY_MS - lat_c)
        lat_c = lat_c + f_out[:, None] * (self.switch_outage_s * 1000.0 / 2.0)  # half the outage on average
        lat_c = np.minimum(lat_c, self.MAX_LATENCY_MS)
        frac_user = np.minimum(1.0, frac_c[assoc, cls]) * (1.0 - f_out[assoc])
        user_ok = ok[assoc]
        user_tput = np.where(user_ok, demand_u * frac_user, 0.0)
        user_lat = np.where(user_ok, lat_c[assoc, cls], 0.0)

        # per-AP aggregates
        ap_tp = np.bincount(assoc, weights=user_tput, minlength=n_ap)
        ap_lat = np.where(counts > 0,
                          np.bincount(assoc, weights=user_lat, minlength=n_ap) / np.maximum(counts, 1.0), 0.0)
        self.ap_throughputs = ap_tp
        self.ap_latencies = ap_lat
        self.ap_util = util
        self.ap_channel_utils = np.minimum(1.0, util)

        sat = user_tput / np.maximum(demand_u, 1e-9)                          # [0,1] satisfaction
        self._user_sat = sat
        qos_ok = (sinr_db >= self.sinr_thresholds[cls]) & (user_lat <= self.latency_sla[cls]) & (sat >= 0.5)

        s1 = np.bincount(assoc, weights=sat, minlength=n_ap)
        s2 = np.bincount(assoc, weights=sat ** 2, minlength=n_ap)
        ap_jain = np.where((counts > 1) & (s2 > 0), s1 ** 2 / np.maximum(counts * s2, 1e-12), 1.0)
        return {"user_tput": user_tput, "user_lat": user_lat, "user_sinr": sinr_db,
                "qos_ok": qos_ok, "ap_jain": ap_jain}

    def _system_metrics(self, net: dict) -> dict:
        sat = self._user_sat
        U = sat.size
        if U == 0:
            return {"mean_user_tput": 0.0, "jfi_users": 1.0, "qos_sat": 1.0, "mean_user_lat": 0.0,
                    "energy_eff": 0.0, "n_switch": int(self.switched.sum()), "mean_util": 0.0}
        jfi = float((sat.sum() ** 2) / (U * (sat ** 2).sum())) if (sat ** 2).sum() > 0 else 1.0
        tx_w = 10 ** (self.ap_powers / 10) / 1000.0
        power_w = (self.AP_STATIC_POWER_W + tx_w / self.PA_EFFICIENCY).sum()
        return {
            "mean_user_tput": float(net["user_tput"].mean()),
            "jfi_users": jfi,
            "qos_sat": float(net["qos_ok"].mean()),
            "mean_user_lat": float(net["user_lat"].mean()),
            "energy_eff": float(net["user_tput"].sum() / max(power_w, 1e-9)),
            "n_switch": int(self.switched.sum()),
            "mean_util": float(self.ap_util[self.ap_user_counts > 0].mean()),
        }

    # ----------------------------------------------------------------- RL API
    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        n_users = int(self.rng.integers(self.n_users_max // 2, self.n_users_max))
        self._generate_users(n_users)
        self.schedule_events = self._generate_schedule_events()
        self.ap_channels = self.rng.integers(0, self.n_channels, self.n_aps)
        self.ap_powers = np.full(self.n_aps, (self.p_min + self.p_max) / 2)
        self.ap_bw_alloc = np.ones((self.n_aps, 3)) / 3
        self.switched = np.zeros(self.n_aps, dtype=bool)
        net = self._compute_network()
        self.last_sys = self._system_metrics(net)
        return self._get_obs()

    def apply_actions(self, actions: dict[int, np.ndarray]) -> None:
        prev = self.ap_channels.copy()
        ids = np.fromiter(actions.keys(), dtype=int)
        A = np.stack([np.asarray(actions[i], dtype=float) for i in ids])
        self.ap_powers[ids] = np.clip(A[:, 0], self.p_min, self.p_max)
        self.ap_channels[ids] = A[:, 1:1 + self.n_channels].argmax(axis=1)
        self.ap_bw_alloc[ids] = quantise_simplex_batch(A[:, 1 + self.n_channels:1 + self.n_channels + 3], self.edca_levels)
        self.switched = self.ap_channels != prev

    def step(self, actions: dict[int, np.ndarray]):
        self.step_count += 1
        self.apply_actions(actions)
        self._simulate_mobility()
        self._apply_schedule_effects()
        net = self._compute_network()
        self.last_sys = self._system_metrics(net)

        rewards, info = {}, {"throughputs": {}, "latencies": {}, "jain": {}, "loads": {}, "util": {},
                             "sys": self.last_sys}
        for i in range(self.n_aps):
            r, ri = self._compute_reward(i, net)
            rewards[i] = r
            info["throughputs"][i] = float(self.ap_throughputs[i])
            info["latencies"][i] = float(self.ap_latencies[i])
            info["jain"][i] = ri["jain"]
            info["loads"][i] = float(self.ap_loads[i])
            info["util"][i] = float(self.ap_util[i])
        done = self.step_count >= self.episode_len
        return self._get_obs(), rewards, done, info

    # --------------------------------------------------------------- dynamics
    def _simulate_mobility(self) -> None:
        # remove expired surge users
        if len(self.user_expire):
            keep = self.user_expire > self.step_count
            if not keep.all():
                self.user_positions = self.user_positions[keep]
                self.user_qos_class = self.user_qos_class[keep]
                self.user_ap_assoc = self.user_ap_assoc[keep]
                self.user_expire = self.user_expire[keep]
        if len(self.user_positions) == 0:
            return
        self.user_positions = self.user_positions + self.rng.normal(0, 2.0, self.user_positions.shape)
        self.user_ap_assoc = self._nearest_ap(self.user_positions)

    def _apply_schedule_effects(self) -> None:
        for event in self.schedule_events:
            tau = abs(event["time_offset"] - self.step_count)
            if tau < 2:
                surge_users = int(event["n_passengers"] * np.exp(-tau ** 2 / 2))
                if event["type"] == "arrival" or self.step_count < event["time_offset"]:
                    self._add_surge_users(event["zone"], surge_users)

    def _add_surge_users(self, zone: int, n_users: int) -> None:
        zone_aps = np.nonzero(self.zone_vec == zone)[0]
        n_users = min(n_users, 100)
        if zone_aps.size == 0 or n_users <= 0:
            return
        aps = self.rng.choice(zone_aps, size=n_users)
        pos = self.ap_positions[aps] + self.rng.normal(0, 10, (n_users, 2))
        self.user_positions = np.vstack([self.user_positions, pos])
        self.user_qos_class = np.append(self.user_qos_class,
                                        self.rng.choice(3, size=n_users, p=self.class_probs))
        self.user_ap_assoc = np.append(self.user_ap_assoc, self._nearest_ap(pos))
        self.user_expire = np.append(self.user_expire,
                                     np.full(n_users, self.step_count + self.USER_LIFETIME_SURGE + 1))

    # ------------------------------------------------------------------ reward
    def _compute_reward(self, ap_id: int, net: dict) -> tuple[float, dict]:
        users = self.user_ap_assoc == ap_id
        n_users = int(users.sum())
        if n_users == 0:
            return 0.0, {"jain": 1.0, "qos_violations": 0}
        r_tp = min(1.0, self.ap_throughputs[ap_id] / 100.0)
        r_lat = -min(1.0, self.ap_latencies[ap_id] / 100.0)
        jain = float(net["ap_jain"][ap_id])
        qos_viol = float((~net["qos_ok"][users]).mean())
        r_qos = -qos_viol
        w = self.rw
        reward = w["tp"] * r_tp + w["lat"] * r_lat + w["fair"] * jain + w["qos"] * r_qos
        return float(reward), {"jain": jain, "qos_violations": qos_viol}

    def _get_obs(self) -> dict[str, Any]:
        per_ap = max(self.n_users_max / self.n_aps, 1)
        h = np.column_stack([
            self.ap_loads,
            self.ap_throughputs / 100.0,
            self.ap_latencies / 500.0,
            self.ap_channel_utils,
            self.ap_user_counts / per_ap,
        ])
        qos = np.zeros((self.n_aps, 3))
        if len(self.user_ap_assoc):
            for c in range(3):
                qos[:, c] = np.bincount(self.user_ap_assoc[self.user_qos_class == c], minlength=self.n_aps)
            qos = qos / np.maximum(qos.sum(axis=1, keepdims=True), 1.0)
        # radio context known to an AP: its channel and what it senses on that channel
        radio = np.zeros((self.n_aps, self.n_channels + 2))
        radio[np.arange(self.n_aps), self.ap_channels] = 1.0
        radio[:, self.n_channels] = self.ap_domain_users / 100.0
        radio[:, self.n_channels + 1] = self.ap_domain_aps / 5.0
        return {
            "radio": radio.astype(np.float32),
            "h": h.astype(np.float32),
            "qos": qos.astype(np.float32),
            "adjacency": self.adjacency.astype(np.float32),
            "loads": self.ap_loads.copy().astype(np.float32),
            "util": self.ap_util.copy().astype(np.float32),
            "true_loads": self.ap_loads.copy().astype(np.float32),
        }

    def get_adjacency_tensor(self) -> torch.Tensor:
        return torch.tensor(self.adjacency, dtype=torch.float32)
