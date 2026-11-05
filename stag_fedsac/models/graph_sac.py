"""
Graph-SAC: batched per-AP Soft Actor-Critic agents with hybrid actions.

Every AP owns an independent actor/critic. For speed, the N independent
networks are stored as stacked tensors (weights of shape [N, in, out]) and
evaluated with batched matrix products; gradients and Adam statistics are
therefore exactly those of N separate networks (the loss is a sum of the N
per-agent losses). Federation operates on slices of these tensors.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6


class BLinear(nn.Module):
    """N independent affine layers: x [N, B, in] -> [N, B, out]."""

    def __init__(self, n: int, d_in: int, d_out: int):
        super().__init__()
        bound = 1.0 / math.sqrt(d_in)
        self.weight = nn.Parameter(torch.empty(n, d_in, d_out).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(n, 1, d_out).uniform_(-bound, bound))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.baddbmm(self.bias, x, self.weight)


class BatchedActor(nn.Module):
    """N personalised SAC actors: federated base + personal head.

    Outputs a hybrid action [power(1), channel(C), bandwidth(3)]:
      * power: reparameterised Gaussian squashed by a sigmoid to [P_min, P_max]
      * channel: straight-through Gumbel-softmax over C orthogonal channels (the one-hot that
        is executed is the one that is stored and fed to the critic; the soft sample carries gradients)
      * bandwidth: Dirichlet over the 3 QoS classes (3-simplex)
    """

    def __init__(self, n: int, state_dim: int, n_channels: int = 11, base_hidden: int = 256,
                 head_hidden: int = 128, p_min: float = 5.0, p_max: float = 23.0):
        super().__init__()
        self.n, self.n_channels, self.p_min, self.p_max = n, n_channels, p_min, p_max
        self.base = nn.ModuleList([BLinear(n, state_dim, base_hidden), BLinear(n, base_hidden, base_hidden)])
        self.head = BLinear(n, base_hidden, head_hidden)
        self.power_mean = BLinear(n, head_hidden, 1)
        self.power_log_std = BLinear(n, head_hidden, 1)
        self.channel_logits = BLinear(n, head_hidden, n_channels)
        self.bw_alpha = BLinear(n, head_hidden, 3)

    def init_default(self, plan: torch.Tensor, channel_logit: float = 4.0) -> None:
        """Start close to the incumbent configuration: channel plan ``plan`` [N], mid power, equal airtime.
        The output layers are scaled down so that the initial policy depends little on the state."""
        with torch.no_grad():
            for layer in (self.power_mean, self.power_log_std, self.channel_logits, self.bw_alpha):
                layer.weight.mul_(0.1)
                layer.bias.zero_()
            self.power_log_std.bias.fill_(-1.5)
            self.bw_alpha.bias.fill_(3.9)          # softplus(3.9)+1 ~ 5 for every class -> mean 1/3
            idx = plan.view(-1, 1, 1).to(self.channel_logits.bias.device)
            self.channel_logits.bias.scatter_(2, idx, channel_logit)

    def base_parameters(self) -> list[nn.Parameter]:
        return [p for m in self.base for p in m.parameters()]

    def base_forward(self, s: torch.Tensor) -> torch.Tensor:
        return F.relu(self.base[1](F.relu(self.base[0](s))))

    def soft_outputs(self, s: torch.Tensor):
        """Expected (soft) policy outputs used for distillation:
        channel probabilities [N,B,C], bandwidth mean [N,B,3], normalised power [N,B,1]."""
        h = F.relu(self.head(self.base_forward(s)))
        ch = F.softmax(self.channel_logits(h), dim=-1)
        alpha = F.softplus(self.bw_alpha(h)) + 1.0
        bw = alpha / alpha.sum(dim=-1, keepdim=True)
        pw = torch.sigmoid(self.power_mean(h))
        return ch, bw, pw

    def forward(self, s: torch.Tensor, deterministic: bool = False, gumbel_tau: float = 1.0):
        """s: [N, B, state_dim] -> (action [N,B,A], log_prob [N,B,1], channel_soft, bw)."""
        h = F.relu(self.head(self.base_forward(s)))
        p_mean = self.power_mean(h)
        p_std = self.power_log_std(h).clamp(LOG_SIG_MIN, LOG_SIG_MAX).exp()
        if deterministic:
            p_raw = p_mean
        else:
            p_dist = Normal(p_mean, p_std)
            p_raw = p_dist.rsample()
        p_sig = torch.sigmoid(p_raw)
        p = p_sig * (self.p_max - self.p_min) + self.p_min
        if deterministic:
            log_prob_p = torch.zeros_like(p)
        else:
            log_prob_p = p_dist.log_prob(p_raw) - torch.log(p_sig * (1.0 - p_sig) + EPSILON)

        ch_logits = self.channel_logits(h)
        if deterministic:
            ch_soft = F.one_hot(ch_logits.argmax(dim=-1), self.n_channels).float()
        else:
            ch_soft = F.gumbel_softmax(ch_logits, tau=gumbel_tau, hard=True)   # straight-through one-hot
        log_prob_ch = (ch_soft * F.log_softmax(ch_logits, dim=-1)).sum(dim=-1, keepdim=True)

        alpha = F.softplus(self.bw_alpha(h)) + 1.0
        if deterministic:
            bw = alpha / alpha.sum(dim=-1, keepdim=True)
            log_prob_bw = torch.zeros_like(log_prob_p)
        else:
            dist = Dirichlet(alpha)
            bw = dist.rsample()
            log_prob_bw = dist.log_prob(bw).unsqueeze(-1)

        action = torch.cat([p, ch_soft, bw], dim=-1)
        return action, log_prob_p + log_prob_ch + log_prob_bw, ch_soft, bw


class BatchedCritic(nn.Module):
    """N double-Q critics."""

    def __init__(self, n: int, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.q1 = nn.ModuleList([BLinear(n, state_dim + action_dim, hidden), BLinear(n, hidden, hidden), BLinear(n, hidden, 1)])
        self.q2 = nn.ModuleList([BLinear(n, state_dim + action_dim, hidden), BLinear(n, hidden, hidden), BLinear(n, hidden, 1)])

    @staticmethod
    def _mlp(layers, x):
        x = F.relu(layers[0](x))
        x = F.relu(layers[1](x))
        return layers[2](x)

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        sa = torch.cat([s, a], dim=-1)
        return self._mlp(self.q1, sa), self._mlp(self.q2, sa)


def per_agent_clip_(params: list[torch.Tensor], max_norm: float) -> None:
    """Clip gradient norm separately for each of the N stacked agents."""
    sq = None
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.pow(2).flatten(1).sum(dim=1)
        sq = g if sq is None else sq + g
    if sq is None:
        return
    scale = (max_norm / (sq.sqrt() + 1e-6)).clamp(max=1.0)
    for p in params:
        if p.grad is not None:
            p.grad.mul_(scale.view(-1, *([1] * (p.grad.dim() - 1))))


class BatchedDDPGActor(nn.Module):
    """N deterministic DDPG actors (baselines). Output in [-1, 1]."""

    def __init__(self, n: int, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.l1, self.l2, self.l3 = BLinear(n, state_dim, hidden), BLinear(n, hidden, hidden), BLinear(n, hidden, action_dim)

    def init_default(self, plan: torch.Tensor, n_channels: int) -> None:
        """Start at the incumbent configuration (round-robin channel plan, mid power, equal airtime)."""
        with torch.no_grad():
            self.l3.weight.mul_(0.1)
            self.l3.bias.zero_()
            target = (2.0 * plan.float() / (n_channels - 1) - 1.0).clamp(-0.97, 0.97)
            self.l3.bias[:, 0, 1] = torch.atanh(target).to(self.l3.bias.device)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.l3(F.relu(self.l2(F.relu(self.l1(s))))))


class BatchedDDPGCritic(nn.Module):
    def __init__(self, n: int, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.l1, self.l2, self.l3 = BLinear(n, state_dim + action_dim, hidden), BLinear(n, hidden, hidden), BLinear(n, hidden, 1)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.l3(F.relu(self.l2(F.relu(self.l1(torch.cat([s, a], dim=-1))))))
