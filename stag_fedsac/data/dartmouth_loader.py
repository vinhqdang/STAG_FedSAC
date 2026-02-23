"""
Dartmouth'18 Campus WiFi Trace — Data loader and preprocessor.
DOI: 10.15783/C7F59T
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DartmouthLoader:
    """Loads and preprocesses Dartmouth campus WiFi traces for ST-GCAT training.

    Domain adaptation: campus buildings → airport terminal zones;
    class periods → flight departure windows.
    """

    def __init__(self, data_dir: str = "data/dartmouth"):
        self.data_dir = data_dir
        self.n_aps = 30  # subset of ~3000 APs
        self.t_history = 48
        self.delta = 6
        self.f_features = 5

    def load_or_generate_synthetic(self) -> dict:
        """Load real data if available, otherwise generate synthetic data
        following Dartmouth statistical properties.

        Returns dict with train/val/test splits.
        """
        real_data_path = os.path.join(self.data_dir, "processed_sessions.csv")

        if os.path.exists(real_data_path):
            return self._load_real_data(real_data_path)
        else:
            print(
                "[DartmouthLoader] Real data not found. "
                "Generating synthetic data with Dartmouth statistical properties..."
            )
            return self._generate_synthetic_data()

    def _load_real_data(self, path: str) -> dict:
        """Load real Dartmouth CSV sessions and aggregate to 5-min intervals."""
        df = pd.read_csv(path, parse_dates=["timestamp"])

        # Group by AP, resample to 5-min intervals
        ap_ids = sorted(df["ap_id"].unique())[:self.n_aps]
        ap_map = {ap: i for i, ap in enumerate(ap_ids)}

        # Create time series
        time_range = pd.date_range(
            df["timestamp"].min(),
            df["timestamp"].max(),
            freq="5min",
        )
        n_steps = len(time_range)

        # Initialize feature matrix [time, N_ap, F]
        features = np.zeros((n_steps, self.n_aps, self.f_features))

        for ap_id in ap_ids:
            ap_df = df[df["ap_id"] == ap_id].set_index("timestamp")
            ap_idx = ap_map[ap_id]

            for t_idx, t in enumerate(time_range):
                window = ap_df[t:t + pd.Timedelta("5min")]
                features[t_idx, ap_idx, 0] = len(window) / 50  # normalized load
                features[t_idx, ap_idx, 4] = len(window)  # user count

        # Normalize
        for f in range(self.f_features):
            p99 = np.percentile(features[:, :, f], 99)
            if p99 > 0:
                features[:, :, f] /= p99
                features[:, :, f] = np.clip(features[:, :, f], 0, 1)

        return self._split_data(features)

    def _generate_synthetic_data(self) -> dict:
        """Generate synthetic WiFi load time series mimicking Dartmouth patterns.

        Properties:
        - Daily periodicity (higher load during 8am-6pm)
        - Weekly patterns (lower on weekends)
        - Class schedule surges (peaks at class start/end)
        - Spatial correlation between nearby APs
        - Random noise component
        """
        n_days = 30  # 1 month of data
        steps_per_day = 288  # 24h * 60min / 5min
        n_steps = n_days * steps_per_day

        features = np.zeros((n_steps, self.n_aps, self.f_features))

        for t in range(n_steps):
            hour = (t % steps_per_day) / steps_per_day * 24
            day_of_week = (t // steps_per_day) % 7

            # Daily pattern: peak during business hours
            daily = 0.3 * np.exp(-((hour - 10) ** 2) / 8) + 0.4 * np.exp(
                -((hour - 14) ** 2) / 10
            )

            # Weekend reduction
            if day_of_week >= 5:
                daily *= 0.4

            # Class schedule surges (on weekdays)
            class_hours = [8, 9, 10, 11, 13, 14, 15, 16]
            surge = 0
            for ch in class_hours:
                tau = hour - ch
                surge += 0.2 * np.exp(-tau ** 2 / 0.1)

            if day_of_week >= 5:
                surge = 0

            for ap in range(self.n_aps):
                # Spatial variation: each AP has different baseline
                ap_bias = 0.1 * np.sin(ap * 0.5) + 0.05 * np.cos(ap * 0.3)

                # Zone-based variation
                zone = ap % 3
                zone_factor = [1.0, 0.8, 1.2][zone]

                base_load = (daily + surge + ap_bias) * zone_factor
                noise = np.random.randn() * 0.05

                load = np.clip(base_load + noise, 0, 1)
                features[t, ap, 0] = load  # normalized load
                features[t, ap, 1] = load * (20 + np.random.randn() * 2)  # throughput
                features[t, ap, 2] = max(5, 50 * load + np.random.randn() * 5) / 500  # latency
                features[t, ap, 3] = min(1, load + np.random.randn() * 0.1)  # channel util
                features[t, ap, 4] = max(0, load * 30 + np.random.randn() * 3) / 50  # user count

        # Normalize each feature to [0, 1]
        for f in range(self.f_features):
            p99 = np.percentile(features[:, :, f], 99)
            if p99 > 0:
                features[:, :, f] /= p99
                features[:, :, f] = np.clip(features[:, :, f], 0, 1)

        return self._split_data(features)

    def _split_data(self, features: np.ndarray) -> dict:
        """Temporal split: 70% train, 15% val, 15% test."""
        n = features.shape[0]
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        return {
            "train": features[:train_end],
            "val": features[train_end:val_end],
            "test": features[val_end:],
            "n_aps": self.n_aps,
        }

    def build_adjacency(self) -> np.ndarray:
        """Build campus building adjacency (synthetic for now)."""
        A = np.zeros((self.n_aps, self.n_aps))
        for i in range(self.n_aps):
            for j in range(i + 1, min(i + 5, self.n_aps)):
                w = 0.8 / (j - i)
                A[i, j] = w
                A[j, i] = w
        return A


class DartmouthDataset(Dataset):
    """PyTorch Dataset for ST-GCAT training on Dartmouth data."""

    def __init__(
        self,
        features: np.ndarray,
        t_history: int = 48,
        delta: int = 6,
    ):
        """
        Args:
            features: [time, N_ap, F] feature matrix
            t_history: historical window length
            delta: prediction horizon
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.t_history = t_history
        self.delta = delta
        self.n_samples = max(0, features.shape[0] - t_history - delta)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        h_start = idx
        h_end = idx + self.t_history
        pred_end = h_end + self.delta

        H = self.features[h_start:h_end]  # [T, N, F]
        H = H.permute(1, 0, 2)  # [N, T, F]

        # Target: future loads
        L_true = self.features[h_end:pred_end, :, 0]  # [delta, N]
        L_true = L_true.permute(1, 0)  # [N, delta]

        return {
            "H": H,
            "L_true": L_true,
        }
