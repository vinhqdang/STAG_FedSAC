"""
MAC-layer helpers for the WiFi simulator.

* ``bianchi_efficiency``: saturation MAC efficiency of IEEE 802.11 DCF as a
  function of the number of contending stations (Bianchi, IEEE JSAC 2000),
  with A-MPDU aggregation. Used to model contention among stations that
  share a collision domain.
* ``quantise_simplex``: maps the continuous 3-simplex bandwidth action onto the
  discrete per-access-category airtime budgets that a real AP can enforce.
"""
from __future__ import annotations

import numpy as np

# 802.11ac OFDM timing (microseconds)
SLOT_US = 9.0
SIFS_US = 16.0
DIFS_US = SIFS_US + 2 * SLOT_US
PHY_HDR_US = 40.0          # preamble + PLCP header (VHT)
BLOCK_ACK_US = 32.0        # block-ack at basic rate
MPDU_BITS = 12000          # 1500-byte MPDU
AMPDU_LEN = 16             # MPDUs per aggregate
CW_MIN = 16
BACKOFF_STAGES = 6         # CWmax = 1024


def _solve_tau_p(n: int, w: int = CW_MIN, m: int = BACKOFF_STAGES) -> tuple[float, float]:
    """Fixed point of Bianchi's coupled equations (tau, p)."""
    if n <= 1:
        return 2.0 / (w + 1), 0.0
    tau = 0.1
    for _ in range(500):
        p = 1.0 - (1.0 - tau) ** (n - 1)
        denom = (1 - 2 * p) * (w + 1) + p * w * (1 - (2 * p) ** m)
        tau_new = 2 * (1 - 2 * p) / denom
        tau_new = float(np.clip(tau_new, 1e-6, 1.0))
        if abs(tau_new - tau) < 1e-10:
            tau = tau_new
            break
        tau = 0.5 * tau + 0.5 * tau_new
    p = 1.0 - (1.0 - tau) ** (n - 1)
    return tau, p


def bianchi_efficiency(n: int, phy_rate_mbps: float = 156.0) -> float:
    """Fraction of airtime delivering payload for n saturated stations."""
    n = max(int(n), 1)
    tau, _ = _solve_tau_p(n)
    p_tr = 1.0 - (1.0 - tau) ** n
    p_s = n * tau * (1.0 - tau) ** (n - 1) / p_tr if p_tr > 0 else 1.0
    payload_bits = MPDU_BITS * AMPDU_LEN
    t_data = PHY_HDR_US + payload_bits / phy_rate_mbps          # us (Mbps == bit/us)
    t_s = t_data + SIFS_US + BLOCK_ACK_US + DIFS_US
    t_c = t_data + DIFS_US
    payload_time = p_tr * p_s * (payload_bits / phy_rate_mbps)
    mean_slot = (1 - p_tr) * SLOT_US + p_tr * p_s * t_s + p_tr * (1 - p_s) * t_c
    return float(payload_time / mean_slot)


def mean_service_time_ms(n: int, phy_rate_mbps: float = 156.0) -> float:
    """Mean time between successful aggregate deliveries in a domain with n
    contenders: payload airtime divided by the Bianchi efficiency (includes
    back-off, collisions and inter-frame spacing)."""
    eff = bianchi_efficiency(n, phy_rate_mbps)
    payload_bits = MPDU_BITS * AMPDU_LEN
    return float((payload_bits / phy_rate_mbps) / max(eff, 1e-6) / 1000.0)


class EfficiencyTable:
    """Pre-computed Bianchi efficiency / service time for n = 0..n_max."""

    def __init__(self, n_max: int = 400, phy_rate_mbps: float = 156.0):
        self.n_max = n_max
        self.eff = np.array([1.0] + [bianchi_efficiency(n, phy_rate_mbps) for n in range(1, n_max + 1)])
        self.svc_ms = np.array([0.0] + [mean_service_time_ms(n, phy_rate_mbps) for n in range(1, n_max + 1)])

    def efficiency(self, n: np.ndarray) -> np.ndarray:
        idx = np.clip(np.rint(n).astype(int), 0, self.n_max)
        return self.eff[idx]

    def service_ms(self, n: np.ndarray) -> np.ndarray:
        idx = np.clip(np.rint(n).astype(int), 0, self.n_max)
        return self.svc_ms[idx]


def quantise_simplex(w: np.ndarray, levels: int = 16, min_levels: int = 1) -> np.ndarray:
    """Largest-remainder quantisation of a simplex vector to multiples of 1/levels.

    Every access category keeps at least ``min_levels`` units so that no queue is
    starved (mirrors the minimum airtime weight accepted by AP schedulers).
    """
    w = np.asarray(w, dtype=float)
    w = np.maximum(w, 0.0)
    s = w.sum()
    w = w / s if s > 0 else np.full_like(w, 1.0 / len(w))
    k = len(w)
    free = levels - k * min_levels
    raw = w * free
    base = np.floor(raw).astype(int)
    rem = free - base.sum()
    order = np.argsort(-(raw - base))
    base[order[:rem]] += 1
    return (base + min_levels) / levels


def quantise_simplex_batch(w: np.ndarray, levels: int = 16, min_levels: int = 1) -> np.ndarray:
    """Row-wise ``quantise_simplex`` for w of shape [n, k]."""
    w = np.maximum(np.asarray(w, dtype=float), 0.0)
    s = w.sum(axis=1, keepdims=True)
    k = w.shape[1]
    w = np.where(s > 0, w / np.maximum(s, 1e-12), 1.0 / k)
    free = levels - k * min_levels
    raw = w * free
    base = np.floor(raw)
    rem = (free - base.sum(axis=1)).astype(int)
    order = np.argsort(-(raw - base), axis=1, kind="stable")
    ranks = np.argsort(order, axis=1, kind="stable")
    base = base + (ranks < rem[:, None])
    return (base + min_levels) / levels
