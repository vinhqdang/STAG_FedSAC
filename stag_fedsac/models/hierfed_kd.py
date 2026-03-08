"""
HierFed-KD: Hierarchical Federated Aggregation with Knowledge Distillation.
Module 3 — Two-level zone/global federation with KD compression.
"""
import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierFedKD:
    """Hierarchical Federated Learning coordinator.

    Level 1: Zone-level quality-weighted FedAvg (every T_local steps)
    Level 2: Global FedProx + KD distillation (every T_global steps)
    """

    def __init__(
        self,
        zone_assignments: dict[int, int],  # ap_id → zone_id
        n_zones: int = 3,
        fedprox_mu: float = 0.01,
        lr_kd: float = 5e-4,
        s_ref_size: int = 100,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            zone_assignments: mapping from AP index to zone index
            n_zones:          number of venue zones
            fedprox_mu:       proximal term weight
            lr_kd:            learning rate for KD distillation
            s_ref_size:       number of reference states for KD
        """
        self.zone_assignments = zone_assignments
        self.n_zones = n_zones
        self.fedprox_mu = fedprox_mu
        self.lr_kd = lr_kd
        self.s_ref_size = s_ref_size
        self.device = device

        # Build zone → AP mapping
        self.zone_aps: dict[int, list[int]] = defaultdict(list)
        for ap_id, zone_id in zone_assignments.items():
            self.zone_aps[zone_id].append(ap_id)

        # Rolling reward tracking for quality weights
        self.reward_history: dict[int, list[float]] = defaultdict(list)
        self.reward_window = 50
        self.baseline_rewards: dict[int, float] = defaultdict(float)

        # Reference state buffer for KD
        self.s_ref: torch.Tensor | None = None

        # Previous global model — used as FedProx anchor.
        # On first round this is None and we fall back to FedAvg.
        self._global_model_prev: dict | None = None

        # Global model state dict (initialized on first aggregation)
        self.global_base_state: dict | None = None

    def update_reward(self, ap_id: int, reward: float) -> None:
        """Track rolling reward for quality-weighted aggregation."""
        self.reward_history[ap_id].append(reward)
        if len(self.reward_history[ap_id]) > self.reward_window:
            self.reward_history[ap_id].pop(0)

    def _compute_quality_weights(self, ap_ids: list[int]) -> dict[int, float]:
        """Compute softmax quality weights based on rolling reward advantage."""
        scores = {}
        for ap_id in ap_ids:
            rewards = self.reward_history[ap_id]
            if len(rewards) > 0:
                avg_reward = sum(rewards) / len(rewards)
                score = avg_reward - self.baseline_rewards[ap_id]
                self.baseline_rewards[ap_id] = avg_reward
            else:
                score = 0.0
            scores[ap_id] = score

        # Softmax
        max_score = max(scores.values()) if scores else 0.0
        exp_scores = {k: torch.exp(torch.tensor(v - max_score)) for k, v in scores.items()}
        total = sum(exp_scores.values())
        return {k: (v / total).item() for k, v in exp_scores.items()}

    def zone_aggregation(
        self, actors: dict[int, nn.Module]
    ) -> None:
        """Level 1: Zone-level quality-weighted FedAvg on base layers.

        Each zone aggregates base parameters from its APs using quality weights.
        Personal heads remain untouched.
        """
        for zone_id, ap_ids in self.zone_aps.items():
            if len(ap_ids) <= 1:
                continue

            weights = self._compute_quality_weights(ap_ids)

            # Aggregate base parameters
            base_state_avg = {}
            for ap_id in ap_ids:
                w = weights[ap_id]
                base_state = {
                    k: v.clone()
                    for k, v in actors[ap_id].base.state_dict().items()
                }
                for k, v in base_state.items():
                    if k not in base_state_avg:
                        base_state_avg[k] = v * w
                    else:
                        base_state_avg[k] += v * w

            # Distribute aggregated base back to zone APs
            for ap_id in ap_ids:
                actors[ap_id].base.load_state_dict(base_state_avg)

    def _fedprox_aggregate(
        self,
        zone_base_states: list[dict],
        mu: float = 0.01,
    ) -> dict:
        """FedProx aggregation across zone models.

        Proximal objective (Li et al., 2020):
            w* = argmin_w  Σ_z ||w - w_z||² + (μ/2)||w - w_prev||²

        Analytic solution:
            w* = (Σ_z w_z + μ·w_prev) / (Z + μ)

        Anchor w_prev is the PREVIOUS global model, NOT the current mean.
        Using anchor=current_mean would make the formula reduce to plain FedAvg
        (since Σw_z + μ*(Σw_z/n) = Σw_z*(1+μ/n) and divides by (n+μ) ≈ n).
        """
        n = len(zone_base_states)
        if n == 0:
            return {}

        # FedAvg mean (baseline / first-round fallback when no prev global exists)
        fedavg_mean = {
            k: sum(s[k].float() for s in zone_base_states) / n
            for k in zone_base_states[0].keys()
        }

        if self._global_model_prev is None:
            # First round: no prior global model → plain FedAvg
            return fedavg_mean

        # FedProx: pull towards previous global model
        global_state = {}
        for k in fedavg_mean.keys():
            if k not in self._global_model_prev:
                global_state[k] = fedavg_mean[k]
                continue
            w_prev = self._global_model_prev[k].to(zone_base_states[0][k].device).float()
            numerator = sum(s[k].float() for s in zone_base_states) + mu * w_prev
            global_state[k] = numerator / (n + mu)

        return global_state

    def global_aggregation(
        self, actors: dict[int, nn.Module],
    ) -> None:
        """Level 2: Global FedProx aggregation + KD distillation.

        Aggregates zone-level models into a global model and distributes
        the global base back to all APs.
        """
        # Collect zone-level base states (one per zone, using first AP as representative)
        zone_base_states = []
        for zone_id in sorted(self.zone_aps.keys()):
            ap_ids = self.zone_aps[zone_id]
            if ap_ids:
                base_state = {
                    k: v.clone()
                    for k, v in actors[ap_ids[0]].base.state_dict().items()
                }
                zone_base_states.append(base_state)

        if not zone_base_states:
            return

        # FedProx aggregation (uses _global_model_prev as anchor if available)
        global_base = self._fedprox_aggregate(zone_base_states)
        # Save current global model as anchor for the NEXT global round
        self._global_model_prev = {k: v.clone().cpu() for k, v in global_base.items()}
        self.global_base_state = global_base

        # Knowledge Distillation step (if reference states available)
        if self.s_ref is not None:
            global_base = self._knowledge_distillation(actors, global_base)

        # Distribute global base to all APs
        for ap_id, actor in actors.items():
            actor.base.load_state_dict(global_base)

    def _knowledge_distillation(
        self,
        actors: dict[int, nn.Module],
        global_base: dict,
    ) -> dict:
        """KD compression: distill soft action distributions from all APs.

        Instead of sharing raw weights, each AP sends soft action distributions
        over shared reference states. ~10× smaller than raw parameter sharing.

        KL direction: KL(local_teacher || global_student) — global student
        minimises divergence from each local teacher.
        """
        template_ap = list(actors.keys())[0]
        global_actor = copy.deepcopy(actors[template_ap])
        global_actor.base.load_state_dict(global_base)
        global_actor.to(self.device)
        global_actor.train()

        optimizer = torch.optim.Adam(global_actor.base.parameters(), lr=self.lr_kd)

        s_ref = self.s_ref.to(self.device)

        # Collect stochastic teacher distributions from local APs (no grad)
        local_ch_logits = []   # list of [M, C] softmax distributions per AP
        local_bw_list   = []   # list of [M, 3] per AP
        with torch.no_grad():
            for actor in actors.values():
                actor.eval()
                # Run stochastically to get SOFT channel distributions
                _, _, l_ch, l_bw = actor(s_ref, deterministic=False)
                # l_ch from gumbel-softmax is already a soft probability vector
                local_ch_logits.append(l_ch.detach())
                local_bw_list.append(l_bw.detach())
                actor.train()

        for _ in range(5):  # few KD steps
            optimizer.zero_grad()
            kd_loss = torch.tensor(0.0, device=self.device)

            # Student forward (with gradients) — stochastic soft distributions
            _, _, g_ch_soft, g_bw_grad = global_actor(s_ref, deterministic=False)
            # g_ch_soft is a soft probability vector from gumbel-softmax
            g_log_probs = torch.log(g_ch_soft + 1e-8)  # log of student probs

            for l_ch, l_bw in zip(local_ch_logits, local_bw_list):
                # KL(teacher || student) = Σ teacher * log(teacher / student)
                ch_kl = F.kl_div(
                    g_log_probs,          # student log-probs
                    l_ch,                  # teacher probs (target)
                    reduction="batchmean",
                    log_target=False,
                )
                bw_mse = F.mse_loss(g_bw_grad, l_bw)
                kd_loss = kd_loss + ch_kl + bw_mse

            kd_loss = kd_loss / len(actors)
            kd_loss.backward()
            optimizer.step()

        return {k: v.clone() for k, v in global_actor.base.state_dict().items()}

    def update_reference_states(
        self, replay_buffers: dict, state_dim: int
    ) -> None:
        """Generate reference states via k-means clustering of replay buffer states."""
        all_states = []
        for buf in replay_buffers.values():
            states = buf.sample_states(min(self.s_ref_size, len(buf)))
            if states is not None:
                all_states.append(states)

        if not all_states:
            return

        all_states = torch.cat(all_states, dim=0)

        # Simple uniform subsampling as approximation to k-means
        if all_states.size(0) > self.s_ref_size:
            indices = torch.randperm(all_states.size(0))[: self.s_ref_size]
            self.s_ref = all_states[indices].to(self.device)
        else:
            self.s_ref = all_states.to(self.device)

    def compute_fedprox_loss(
        self, actor: nn.Module
    ) -> torch.Tensor:
        """Compute FedProx proximal term: μ/2 · ||θ - θ_global||²."""
        if self.global_base_state is None:
            return torch.tensor(0.0, device=self.device)

        prox_loss = torch.tensor(0.0, device=self.device)
        for (name, param), (_, global_param) in zip(
            actor.base.named_parameters(),
            self.global_base_state.items(),
        ):
            global_param = global_param.to(self.device)
            prox_loss += (param - global_param).pow(2).sum()

        return (self.fedprox_mu / 2) * prox_loss
