"""
Lagrangian — Adaptive Lagrange multiplier updates for fairness constraints.
"""
import torch


class LagrangianManager:
    """Manages adaptive Lagrange multipliers for capacity and fairness constraints.

    Updates λ_i ← max(0, λ_i + ρ · (Load_i - L_max))  per AP.
    """

    def __init__(
        self,
        n_aps: int,
        l_max: float = 0.90,
        rho: float = 0.01,
        device: torch.device = torch.device("cpu"),
    ):
        self.n_aps = n_aps
        self.l_max = l_max
        self.rho = rho
        self.device = device
        self.lambdas = torch.zeros(n_aps, device=device)

    def update(self, loads: torch.Tensor) -> None:
        """Update multipliers based on current AP loads.

        Args:
            loads: [N_ap] — current load fraction per AP
        """
        constraint_violation = loads - self.l_max
        self.lambdas = torch.clamp(
            self.lambdas + self.rho * constraint_violation, min=0.0
        )

    def get_penalty(self, ap_id: int, load: float) -> float:
        """Compute Lagrangian penalty for a specific AP."""
        return -self.lambdas[ap_id].item() * max(0.0, load - self.l_max)

    def get_penalties_batch(self, loads: torch.Tensor) -> torch.Tensor:
        """Compute penalties for all APs.

        Args:
            loads: [batch, N_ap] or [N_ap]
        Returns:
            penalties: same shape as loads
        """
        violations = torch.clamp(loads - self.l_max, min=0.0)
        return -self.lambdas * violations
