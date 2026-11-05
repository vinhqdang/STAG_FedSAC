"""
Batched replay buffer: one independent FIFO buffer per AP, stored as stacked
tensors on the compute device. States are kept in float16 to bound memory.
"""
from __future__ import annotations

import torch


class BatchedReplayBuffer:
    def __init__(self, n_agents: int, capacity: int, state_dim: int, action_dim: int,
                 device: torch.device):
        self.n, self.cap, self.device = n_agents, capacity, device
        self.s = torch.zeros(n_agents, capacity, state_dim, dtype=torch.float16, device=device)
        self.s2 = torch.zeros(n_agents, capacity, state_dim, dtype=torch.float16, device=device)
        self.a = torch.zeros(n_agents, capacity, action_dim, device=device)
        self.r = torch.zeros(n_agents, capacity, 1, device=device)
        self.d = torch.zeros(n_agents, capacity, 1, device=device)
        self.ptr = 0
        self.size = 0
        self._ar = torch.arange(n_agents, device=device).unsqueeze(1)

    def __len__(self) -> int:
        return self.size

    def add(self, s, a, r, s2, d) -> None:
        """s,s2: [N,S]; a: [N,A]; r: [N]; d: float."""
        i = self.ptr
        self.s[:, i] = s.to(torch.float16)
        self.s2[:, i] = s2.to(torch.float16)
        self.a[:, i] = a
        self.r[:, i, 0] = r
        self.d[:, i, 0] = float(d)
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch: int, pooled: bool = False):
        """Returns tensors of shape [N, B, .]. ``pooled``: each agent draws from
        the union of all agents' experience (centralised comparator)."""
        t = torch.randint(0, self.size, (self.n, batch), device=self.device)
        who = (torch.randint(0, self.n, (self.n, batch), device=self.device) if pooled else self._ar.expand(-1, batch))
        return (self.s[who, t].float(), self.a[who, t], self.r[who, t], self.s2[who, t].float(), self.d[who, t])

    def sample_states(self, m: int) -> torch.Tensor:
        t = torch.randint(0, self.size, (m,), device=self.device)
        return self.s[:, t].float()
