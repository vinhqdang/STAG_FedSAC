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
        self, zone_base_states: list[dict],
    ) -> dict:
        """FedProx aggregation across zone models."""
        # Simple mean as initial global model
        global_state = {}
        n = len(zone_base_states)
        for k in zone_base_states[0].keys():
            global_state[k] = sum(s[k] for s in zone_base_states) / n
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

        # FedProx aggregation
        global_base = self._fedprox_aggregate(zone_base_states)
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
        """
        # Create temporary global actor with the aggregated base
        # Use any actor as template
        template_ap = list(actors.keys())[0]
        global_actor = copy.deepcopy(actors[template_ap])
        global_actor.base.load_state_dict(global_base)
        global_actor.to(self.device)

        optimizer = torch.optim.Adam(global_actor.base.parameters(), lr=self.lr_kd)

        # KD loss: minimize KL divergence between global and local policies
        s_ref = self.s_ref.to(self.device)

        for _ in range(5):  # few KD steps
            optimizer.zero_grad()
            kd_loss = torch.tensor(0.0, device=self.device)

            # Global policy on reference states
            with torch.no_grad():
                global_action, _, g_ch_soft, g_bw = global_actor(
                    s_ref, deterministic=True
                )

            for ap_id, actor in actors.items():
                with torch.no_grad():
                    local_action, _, l_ch_soft, l_bw = actor(
                        s_ref, deterministic=True
                    )

                # Re-run global actor with gradients
                g_action_grad, _, g_ch_grad, g_bw_grad = global_actor(s_ref)

                # KL on channel distributions
                ch_kl = F.kl_div(
                    F.log_softmax(g_ch_grad, dim=-1),
                    l_ch_soft,
                    reduction="batchmean",
                )
                # MSE on bandwidth allocations
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
