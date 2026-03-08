"""
Graph-SAC: Federated Soft Actor-Critic Agent with Hybrid Actions.
Module 2 — Per-AP SAC agent with graph-embedded state representation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Dirichlet


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6


class SACActorNetwork(nn.Module):
    """SAC actor outputting hybrid continuous-discrete actions.

    Outputs:
        - Power:     reparameterized Gaussian → sigmoid → [P_min, P_max]
        - Channel:   Gumbel-Softmax (differentiable discrete)
        - Bandwidth: Dirichlet (simplex-constrained across QoS classes)
    """

    def __init__(
        self,
        state_dim: int,
        n_channels: int = 11,
        hidden: int = 256,
        p_min: float = 5.0,
        p_max: float = 23.0,
    ):
        super().__init__()
        self.p_min = p_min
        self.p_max = p_max
        self.n_channels = n_channels

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.power_mean = nn.Linear(hidden, 1)
        self.power_log_std = nn.Linear(hidden, 1)
        self.channel_logits = nn.Linear(hidden, n_channels)
        self.bw_alpha = nn.Linear(hidden, 3)  # Dirichlet concentrations

    def forward(
        self, s: torch.Tensor, deterministic: bool = False, gumbel_tau: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (action, log_prob, channel_soft, bw)
            action: concatenated [power(1), channel_soft(C), bw(3)]
            log_prob: scalar log probability
        """
        h = self.shared(s)

        # ── Power: reparameterized Gaussian ──
        p_mean = self.power_mean(h)
        p_log_std = self.power_log_std(h).clamp(LOG_SIG_MIN, LOG_SIG_MAX)
        p_std = p_log_std.exp()

        if deterministic:
            p_raw = p_mean
        else:
            p_dist = Normal(p_mean, p_std)
            p_raw = p_dist.rsample()

        # Squash to [P_min, P_max]  via sigmoid
        p_sig = torch.sigmoid(p_raw)
        p = p_sig * (self.p_max - self.p_min) + self.p_min
        # Log-prob with correct sigmoid Jacobian: d/dx sigmoid(x) = sigmoid(x)*(1-sigmoid(x))
        if deterministic:
            log_prob_p = torch.zeros_like(p)
        else:
            log_prob_p = p_dist.log_prob(p_raw) - torch.log(
                p_sig * (1.0 - p_sig) + EPSILON
            )
            log_prob_p = log_prob_p.sum(dim=-1, keepdim=True)

        # ── Channel: Gumbel-Softmax ──
        ch_logits = self.channel_logits(h)
        if deterministic:
            ch_soft = F.one_hot(
                ch_logits.argmax(dim=-1), self.n_channels
            ).float()
        else:
            ch_soft = F.gumbel_softmax(ch_logits, tau=gumbel_tau, hard=False)
        log_prob_ch = F.log_softmax(ch_logits, dim=-1)
        log_prob_ch = (ch_soft * log_prob_ch).sum(dim=-1, keepdim=True)

        # ── Bandwidth: Dirichlet (simplex) ──
        bw_alpha = F.softplus(self.bw_alpha(h)) + 1.0  # ensure > 1
        bw_dist = Dirichlet(bw_alpha)
        if deterministic:
            bw = bw_alpha / bw_alpha.sum(dim=-1, keepdim=True)
        else:
            bw = bw_dist.rsample()
        log_prob_bw = bw_dist.log_prob(bw).unsqueeze(-1)

        # Total log-prob
        log_prob = log_prob_p + log_prob_ch + log_prob_bw

        # Concatenate action
        action = torch.cat([p, ch_soft, bw], dim=-1)
        return action, log_prob, ch_soft, bw


class SACCriticNetwork(nn.Module):
    """Double-Q critic for SAC."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)


class PersonalizedSACActor(nn.Module):
    """SAC actor with shared base (federated) and personal head (local).

    The base layers participate in federation, the head layers stay local
    to capture zone-specific traffic patterns.
    """

    def __init__(
        self,
        state_dim: int,
        n_channels: int = 11,
        base_hidden: int = 256,
        head_hidden: int = 128,
        p_min: float = 5.0,
        p_max: float = 23.0,
    ):
        super().__init__()
        self.p_min = p_min
        self.p_max = p_max
        self.n_channels = n_channels

        # Shared base (federated)
        self.base = nn.Sequential(
            nn.Linear(state_dim, base_hidden),
            nn.ReLU(),
            nn.Linear(base_hidden, base_hidden),
            nn.ReLU(),
        )
        # Personal head (zone-specific, never federated)
        self.head = nn.Sequential(
            nn.Linear(base_hidden, head_hidden),
            nn.ReLU(),
        )
        self.power_mean = nn.Linear(head_hidden, 1)
        self.power_log_std = nn.Linear(head_hidden, 1)
        self.channel_logits = nn.Linear(head_hidden, n_channels)
        self.bw_alpha = nn.Linear(head_hidden, 3)

    def get_base_params(self) -> list:
        """Return parameters of federated base layers."""
        return list(self.base.parameters())

    def get_head_params(self) -> list:
        """Return parameters of personal head layers."""
        params = list(self.head.parameters())
        params += list(self.power_mean.parameters())
        params += list(self.power_log_std.parameters())
        params += list(self.channel_logits.parameters())
        params += list(self.bw_alpha.parameters())
        return params

    def forward(
        self, s: torch.Tensor, deterministic: bool = False, gumbel_tau: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.head(self.base(s))

        # Power
        p_mean = self.power_mean(h)
        p_log_std = self.power_log_std(h).clamp(LOG_SIG_MIN, LOG_SIG_MAX)
        p_std = p_log_std.exp()

        if deterministic:
            p_raw = p_mean
        else:
            p_dist = Normal(p_mean, p_std)
            p_raw = p_dist.rsample()

        p_sig = torch.sigmoid(p_raw)
        p = p_sig * (self.p_max - self.p_min) + self.p_min

        if deterministic:
            log_prob_p = torch.zeros_like(p_mean)
        else:
            log_prob_p = p_dist.log_prob(p_raw) - torch.log(
                p_sig * (1.0 - p_sig) + EPSILON
            )
            log_prob_p = log_prob_p.sum(dim=-1, keepdim=True)

        # Channel
        ch_logits = self.channel_logits(h)
        if deterministic:
            ch_soft = F.one_hot(
                ch_logits.argmax(dim=-1), self.n_channels
            ).float()
        else:
            ch_soft = F.gumbel_softmax(ch_logits, tau=gumbel_tau, hard=False)
        log_prob_ch = (ch_soft * F.log_softmax(ch_logits, dim=-1)).sum(
            dim=-1, keepdim=True
        )

        # Bandwidth
        bw_alpha = F.softplus(self.bw_alpha(h)) + 1.0
        bw_dist = Dirichlet(bw_alpha)
        if deterministic:
            bw = bw_alpha / bw_alpha.sum(dim=-1, keepdim=True)
            log_prob_bw = torch.zeros(s.size(0), 1, device=s.device)
        else:
            bw = bw_dist.rsample()
            log_prob_bw = bw_dist.log_prob(bw).unsqueeze(-1)

        log_prob = log_prob_p + log_prob_ch + log_prob_bw
        action = torch.cat([p, ch_soft, bw], dim=-1)
        return action, log_prob, ch_soft, bw
