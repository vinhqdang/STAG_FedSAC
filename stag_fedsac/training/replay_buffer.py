"""
Replay Buffer — Per-AP prioritized experience replay.
"""
import torch
import numpy as np


class ReplayBuffer:
    """Simple per-AP experience replay buffer with uniform sampling."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        # Pre-allocate tensors
        self.states = torch.zeros(capacity, state_dim, dtype=torch.float32)
        self.actions = torch.zeros(capacity, action_dim, dtype=torch.float32)
        self.rewards = torch.zeros(capacity, 1, dtype=torch.float32)
        self.next_states = torch.zeros(capacity, state_dim, dtype=torch.float32)
        self.dones = torch.zeros(capacity, 1, dtype=torch.float32)

        # Additional fields for joint training
        self.h_obs = None  # local observations [F]
        self.qos_obs = None  # QoS fractions [3]
        self.sched_obs = None  # schedule features [D_s]

    def store(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        self.states[self.ptr] = state.detach().cpu()
        self.actions[self.ptr] = action.detach().cpu() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.float32)
        self.rewards[self.ptr, 0] = float(reward)
        self.next_states[self.ptr] = next_state.detach().cpu()
        self.dones[self.ptr, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple:
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices].to(self.device),
            self.actions[indices].to(self.device),
            self.rewards[indices].to(self.device),
            self.next_states[indices].to(self.device),
            self.dones[indices].to(self.device),
        )

    def sample_states(self, n: int) -> torch.Tensor | None:
        """Sample n states for reference state generation."""
        if self.size == 0:
            return None
        n = min(n, self.size)
        indices = np.random.randint(0, self.size, size=n)
        return self.states[indices]

    def __len__(self) -> int:
        return self.size
