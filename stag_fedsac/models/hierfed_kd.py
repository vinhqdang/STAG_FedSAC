"""
HierFed-KD: hierarchical federation of the per-AP actors.

Federated parameters: the actor *base* layers. Heads (personalisation) and the
critics never leave the AP.

Modes
-----
none      isolated agents (no sharing)
flat      single-level FedAvg over all APs (cloud server), every T_local steps
zone      Level 1 only: quality-weighted FedAvg inside each venue zone
hier_avg  Level 1 + weight-based global FedProx (zone models exchanged as weights)
hier_kd   Level 1 + proximal federated distillation across zones: zones exchange
          soft policy outputs on a public reference set instead of weights

``CommMeter`` counts the bytes of every message so that communication claims
are measured from the actual tensors, not asserted.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

BYTES = 4  # float32


class CommMeter:
    def __init__(self):
        self.intra_zone = 0      # AP <-> zone controller (local network)
        self.inter_zone = 0      # zone controller <-> venue/cloud server (wide area)
        self.reference_set = 0   # public reference states shipped to the zones

    def total_inter(self) -> int:
        return self.inter_zone + self.reference_set

    def as_dict(self) -> dict:
        return dict(intra_zone=self.intra_zone, inter_zone=self.inter_zone,
                    reference_set=self.reference_set, wan_total=self.total_inter())


class Federation:
    def __init__(self, zone_vec: np.ndarray, mode: str = "hier_kd", mu: float = 0.01,
                 lr_kd: float = 5e-4, kd_steps: int = 20, n_channels: int = 11,
                 m_ref: int = 100, quality_temp: float = 0.1, device: torch.device = torch.device("cpu")):
        assert mode in ("none", "flat", "zone", "hier_avg", "hier_kd")
        self.mode, self.mu, self.lr_kd, self.kd_steps = mode, mu, lr_kd, kd_steps
        self.n_channels, self.m_ref, self.quality_temp, self.device = n_channels, m_ref, quality_temp, device
        self.zone_vec = zone_vec
        self.n = len(zone_vec)
        self.zones = [torch.tensor(np.nonzero(zone_vec == z)[0], device=device) for z in sorted(set(zone_vec.tolist()))]
        self.ema = torch.zeros(self.n, device=device)
        self.anchor: list[torch.Tensor] | None = None     # last received base (proximal anchor)
        self.global_prev: list[torch.Tensor] | None = None
        self.meter = CommMeter()

    @property
    def uses_prox(self) -> bool:
        return self.mode in ("hier_avg", "hier_kd")

    # -- bookkeeping --------------------------------------------------------------
    def update_reward(self, r: torch.Tensor) -> None:
        self.ema = 0.95 * self.ema + 0.05 * r

    @staticmethod
    def _n_params(actor) -> int:
        return int(sum(p[0].numel() for p in actor.base_parameters()))

    def _weights(self, idx: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.ema[idx] / self.quality_temp, dim=0)

    def _snapshot(self, actor) -> None:
        self.anchor = [p.detach().clone() for p in actor.base_parameters()]

    # -- level 1 ------------------------------------------------------------------
    @torch.no_grad()
    def zone_round(self, actor) -> None:
        if self.mode == "none":
            return
        if self.mode == "flat":
            idx = torch.arange(self.n, device=self.device)
            for p in actor.base_parameters():
                p.data[idx] = p.data[idx].mean(dim=0, keepdim=True).expand_as(p.data[idx])
            self.meter.inter_zone += 2 * self.n * self._n_params(actor) * BYTES
            self._snapshot(actor)
            return
        for idx in self.zones:
            w = self._weights(idx)
            for p in actor.base_parameters():
                shape = (-1,) + (1,) * (p.dim() - 1)
                avg = (p.data[idx] * w.view(shape)).sum(dim=0, keepdim=True)
                p.data[idx] = avg.expand_as(p.data[idx])
            self.meter.intra_zone += 2 * len(idx) * self._n_params(actor) * BYTES
        self._snapshot(actor)

    # -- level 2 ------------------------------------------------------------------
    @torch.no_grad()
    def global_round_avg(self, actor) -> None:
        zone_means = [[p.data[idx[0]].clone() for p in actor.base_parameters()] for idx in self.zones]
        Z = len(zone_means)
        new = []
        for k, p in enumerate(actor.base_parameters()):
            s = sum(zm[k] for zm in zone_means)
            if self.global_prev is None:
                new.append(s / Z)
            else:
                new.append((s + self.mu * self.global_prev[k]) / (Z + self.mu))
        self.global_prev = [t.clone() for t in new]
        for k, p in enumerate(actor.base_parameters()):
            p.data[:] = new[k].unsqueeze(0).expand_as(p.data)
        self.meter.inter_zone += 2 * Z * self._n_params(actor) * BYTES
        self._snapshot(actor)

    def global_round_kd(self, actor, s_ref: torch.Tensor) -> None:
        """Zones exchange soft outputs on the public reference set s_ref [M, S].

        Each zone controller averages its APs' outputs (zone teacher); the server
        averages the zone teachers (the closed-form minimiser of the KD objective
        sum_z KL(pi_z || pi) + MSE(w_z, w)) and returns the consensus; every AP
        distils its base layers toward the consensus, with a proximal term
        (mu/2)||theta - theta_before||^2 that keeps it close to its own model."""
        N, M = self.n, s_ref.shape[0]
        sr = s_ref.unsqueeze(0).expand(N, -1, -1)
        with torch.no_grad():
            ch, bw, pw = actor.soft_outputs(sr)
            t_ch = torch.stack([ch[idx].mean(dim=0) for idx in self.zones]).mean(dim=0)
            t_bw = torch.stack([bw[idx].mean(dim=0) for idx in self.zones]).mean(dim=0)
            t_pw = torch.stack([pw[idx].mean(dim=0) for idx in self.zones]).mean(dim=0)
        payload = M * (self.n_channels + 3 + 1) * BYTES
        self.meter.inter_zone += 2 * len(self.zones) * payload
        self.meter.intra_zone += 2 * N * payload
        base = actor.base_parameters()
        before = [p.detach().clone() for p in base]
        opt = torch.optim.Adam(base, lr=self.lr_kd)
        for _ in range(self.kd_steps):
            c, b, w = actor.soft_outputs(sr)
            kl = F.kl_div(torch.log(c + 1e-8), t_ch.unsqueeze(0).expand_as(c), reduction="none").sum(-1).mean(dim=1)
            mse = (b - t_bw.unsqueeze(0)).pow(2).sum(-1).mean(dim=1) + (w - t_pw.unsqueeze(0)).pow(2).sum(-1).mean(dim=1)
            prox = sum((p - q).pow(2).flatten(1).sum(dim=1) for p, q in zip(base, before))
            loss = (kl + mse + 0.5 * self.mu * prox).sum()
            grads = torch.autograd.grad(loss, base)
            for p, g in zip(base, grads):
                p.grad = g
            opt.step()
        for p in base:
            p.grad = None
        self._snapshot(actor)

    def ship_reference_set(self, s_ref: torch.Tensor) -> None:
        self.meter.reference_set += len(self.zones) * s_ref.numel() * BYTES

    # -- local proximal regulariser -------------------------------------------------
    def prox_loss(self, actor) -> torch.Tensor:
        """sum over APs of (mu/2)||theta_i - anchor||^2 (zero until the first round)."""
        if not self.uses_prox or self.anchor is None:
            return torch.zeros((), device=self.device)
        return 0.5 * self.mu * sum((p - a).pow(2).sum() for p, a in zip(actor.base_parameters(), self.anchor))
