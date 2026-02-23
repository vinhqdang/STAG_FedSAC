"""
Graph Builder — Interference graph construction from WACA data or simulation.
"""
import numpy as np
import torch


class InterferenceGraphBuilder:
    """Builds AP interference graphs from RSSI data or AP positions."""

    def __init__(
        self,
        n_aps: int,
        rssi_threshold: float = -85.0,  # dBm
        max_interference_dist: float = 80.0,  # meters
    ):
        self.n_aps = n_aps
        self.rssi_threshold = rssi_threshold
        self.max_dist = max_interference_dist

    def build_from_positions(
        self, ap_positions: np.ndarray, tx_power: float = 20.0
    ) -> np.ndarray:
        """Build interference graph from AP positions using path loss model.

        Args:
            ap_positions: [N, 2] AP coordinates
            tx_power: transmit power in dBm

        Returns:
            A: [N, N] weighted adjacency matrix
        """
        N = len(ap_positions)
        A = np.zeros((N, N))

        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(ap_positions[i] - ap_positions[j])
                if dist < self.max_dist:
                    # Simplified indoor path loss
                    rssi = tx_power - (38.46 + 20 * np.log10(max(dist, 1.0)))
                    if rssi > self.rssi_threshold:
                        # Weight: normalized by threshold range
                        w = (rssi - self.rssi_threshold) / (
                            tx_power - self.rssi_threshold
                        )
                        w = np.clip(w, 0, 1)
                        A[i, j] = w
                        A[j, i] = w

        return A

    def build_from_rssi_matrix(
        self, rssi_matrix: np.ndarray
    ) -> np.ndarray:
        """Build interference graph from pairwise RSSI measurements (WACA data).

        Args:
            rssi_matrix: [N, N] RSSI measurements in dBm

        Returns:
            A: [N, N] weighted adjacency matrix
        """
        mask = rssi_matrix > self.rssi_threshold
        # Normalize to [0, 1]
        A = np.zeros_like(rssi_matrix)
        valid = mask & (rssi_matrix < 0)  # valid RSSI values
        if valid.any():
            A[valid] = (rssi_matrix[valid] - self.rssi_threshold) / (
                0 - self.rssi_threshold
            )
            A = np.clip(A, 0, 1)

        # Make symmetric and zero diagonal
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)
        return A

    def build_from_channel_overlap(
        self,
        ap_channels: np.ndarray,
        ap_positions: np.ndarray,
    ) -> np.ndarray:
        """Build interference graph considering channel overlap.

        Adjacent channels in 2.4GHz overlap. Interference is channel-dependent.

        Args:
            ap_channels: [N] channel assignments
            ap_positions: [N, 2] positions

        Returns:
            A: [N, N] weighted adjacency matrix
        """
        N = len(ap_channels)
        A_pos = self.build_from_positions(ap_positions)

        # Channel overlap factor
        A = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                if A_pos[i, j] > 0:
                    ch_diff = abs(int(ap_channels[i]) - int(ap_channels[j]))
                    # In 2.4GHz, channels need 5-channel separation
                    if ch_diff < 5:
                        overlap_factor = 1.0 - ch_diff / 5.0
                        A[i, j] = A_pos[i, j] * overlap_factor
                        A[j, i] = A[i, j]

        return A

    def to_tensor(self, A: np.ndarray) -> torch.Tensor:
        """Convert adjacency matrix to PyTorch tensor."""
        return torch.tensor(A, dtype=torch.float32)
