"""
WACA WiFi All-Channel Analyzer — Camp Nou data loader.
DOI: 10.5281/zenodo.3960029
"""
import os
import glob
import numpy as np
import pandas as pd


class WACALoader:
    """Loads WACA Camp Nou all-channel RSSI data for interference calibration.

    Provides pairwise RSSI-based interference scores between channel pairs
    for parameterizing the AP interference graph.
    """

    def __init__(self, data_dir: str = "data/waca"):
        self.data_dir = data_dir
        self.n_channels_24 = 14  # 2.4 GHz channels
        self.n_channels_5 = 24  # 5 GHz channels

    def load_or_generate_synthetic(self) -> dict:
        """Load real WACA data or generate synthetic interference data."""
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))

        if csv_files:
            return self._load_real_data(csv_files)
        else:
            print(
                "[WACALoader] Real data not found. "
                "Generating synthetic channel interference data..."
            )
            return self._generate_synthetic_data()

    def _load_real_data(self, csv_files: list[str]) -> dict:
        """Load real WACA CSV files with per-channel RSSI measurements."""
        all_rssi = []
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                all_rssi.append(df)
            except Exception as e:
                print(f"  Warning: Could not load {f}: {e}")

        if not all_rssi:
            return self._generate_synthetic_data()

        # Process RSSI time series
        df = pd.concat(all_rssi, ignore_index=True)

        # Extract channel RSSI columns
        rssi_cols = [
            c for c in df.columns
            if "rssi" in c.lower() or "channel" in c.lower()
        ]

        result = {
            "raw_data": df,
            "rssi_columns": rssi_cols,
            "n_measurements": len(df),
        }

        # Compute interference matrix if possible
        if len(rssi_cols) >= 2:
            rssi_matrix = df[rssi_cols].values
            result["channel_interference"] = self._compute_channel_interference(
                rssi_matrix
            )

        return result

    def _generate_synthetic_data(self) -> dict:
        """Generate synthetic channel interference data mimicking Camp Nou.

        Camp Nou: 90,000 fans, extreme WiFi density.
        """
        n_channels = self.n_channels_24 + self.n_channels_5
        n_measurements = 1000

        # Generate RSSI time series per channel
        rssi_data = np.zeros((n_measurements, n_channels))

        for ch in range(n_channels):
            # Base RSSI level (lower for 5GHz)
            base = -50 if ch < self.n_channels_24 else -60
            # Crowd-induced variation
            crowd_effect = 10 * np.sin(
                2 * np.pi * np.arange(n_measurements) / 200
            )
            noise = np.random.randn(n_measurements) * 5
            rssi_data[:, ch] = base + crowd_effect + noise

        # Channel interference matrix
        interference = self._compute_channel_interference_synthetic(n_channels)

        return {
            "rssi_data": rssi_data,
            "channel_interference": interference,
            "n_channels_24": self.n_channels_24,
            "n_channels_5": self.n_channels_5,
            "n_measurements": n_measurements,
        }

    def _compute_channel_interference(
        self, rssi_matrix: np.ndarray
    ) -> np.ndarray:
        """Compute pairwise channel interference from RSSI correlations."""
        n_ch = rssi_matrix.shape[1]
        interference = np.zeros((n_ch, n_ch))

        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                # Pearson correlation as interference proxy
                valid = ~(np.isnan(rssi_matrix[:, i]) | np.isnan(rssi_matrix[:, j]))
                if valid.sum() > 10:
                    corr = np.corrcoef(rssi_matrix[valid, i], rssi_matrix[valid, j])[0, 1]
                    interference[i, j] = max(0, corr)
                    interference[j, i] = interference[i, j]

        return interference

    def _compute_channel_interference_synthetic(
        self, n_channels: int
    ) -> np.ndarray:
        """Generate realistic channel interference matrix.

        In 2.4GHz band: adjacent channels overlap significantly.
        In 5GHz band: less overlap, more independent channels.
        """
        W = np.zeros((n_channels, n_channels))

        # 2.4 GHz: channels 1-14, need 5-channel separation for no overlap
        for i in range(self.n_channels_24):
            for j in range(i + 1, self.n_channels_24):
                ch_diff = abs(i - j)
                if ch_diff < 5:
                    overlap = 1.0 - ch_diff / 5.0
                    W[i, j] = overlap
                    W[j, i] = overlap

        # 5 GHz: channels are mostly independent (20MHz spacing)
        offset = self.n_channels_24
        for i in range(self.n_channels_5):
            for j in range(i + 1, self.n_channels_5):
                if abs(i - j) == 1:
                    W[offset + i, offset + j] = 0.1  # minimal adjacent overlap
                    W[offset + j, offset + i] = 0.1

        return W

    def get_interference_weights(self, data: dict) -> np.ndarray:
        """Get normalized interference weight matrix for SAC reward."""
        if "channel_interference" in data:
            W = data["channel_interference"]
            # Normalize to [0, 1]
            if W.max() > 0:
                W = W / W.max()
            return W
        return self._compute_channel_interference_synthetic(
            self.n_channels_24 + self.n_channels_5
        )
