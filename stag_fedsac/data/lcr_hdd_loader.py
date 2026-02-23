"""
LCR HDD 5G High Density Demand Dataset — Data loader.
DOI: 10.1038/s41597-025-06282-0
"""
import os
import glob
import numpy as np
import pandas as pd


class LCRHDDLoader:
    """Loads and preprocesses the LCR HDD ACC Arena dataset.

    Provides user positions, QoS traffic types, SINR, throughput
    for DRL environment parameterization.
    """

    def __init__(self, data_dir: str = "data/lcr_hdd"):
        self.data_dir = data_dir

    def load_or_generate_synthetic(self) -> dict:
        """Load real LCR HDD data or generate synthetic version."""
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))

        if csv_files:
            return self._load_real_data(csv_files)
        else:
            print(
                "[LCRHDDLoader] Real data not found. "
                "Generating synthetic high-density venue data..."
            )
            return self._generate_synthetic_data()

    def _load_real_data(self, csv_files: list[str]) -> dict:
        """Load real LCR HDD CSV files."""
        all_data = []
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                all_data.append(df)
            except Exception as e:
                print(f"  Warning: Could not load {f}: {e}")

        if not all_data:
            return self._generate_synthetic_data()

        df = pd.concat(all_data, ignore_index=True)

        # Extract relevant columns
        result = {
            "user_positions": None,
            "traffic_types": None,
            "sinr": None,
            "throughput": None,
            "ru_association": None,
            "n_samples": len(df),
        }

        # Map column names (varies by dataset version)
        pos_cols = [c for c in df.columns if "pos" in c.lower() or c in ["x", "y"]]
        if len(pos_cols) >= 2:
            result["user_positions"] = df[pos_cols[:2]].values

        sinr_cols = [c for c in df.columns if "sinr" in c.lower()]
        if sinr_cols:
            result["sinr"] = df[sinr_cols[0]].values

        tp_cols = [c for c in df.columns if "throughput" in c.lower() or "tp" in c.lower()]
        if tp_cols:
            result["throughput"] = df[tp_cols[0]].values

        return result

    def _generate_synthetic_data(self) -> dict:
        """Generate synthetic high-density venue data mimicking ACC Arena.

        ACC Arena: indoor, up to 12,000 users, 33 Radio Units.
        """
        n_users = 3000
        n_rus = 33
        arena_size = (100, 80)  # meters

        # User positions (clustered around seating areas)
        positions = np.zeros((n_users, 2))
        n_clusters = 6
        for i in range(n_users):
            cluster = i % n_clusters
            cx = (cluster % 3) * arena_size[0] / 3 + arena_size[0] / 6
            cy = (cluster // 3) * arena_size[1] / 2 + arena_size[1] / 4
            positions[i, 0] = cx + np.random.randn() * 15
            positions[i, 1] = cy + np.random.randn() * 10

        positions[:, 0] = np.clip(positions[:, 0], 0, arena_size[0])
        positions[:, 1] = np.clip(positions[:, 1], 0, arena_size[1])

        # Traffic types: 20% VoIP, 30% Video, 50% BestEffort
        traffic_types = np.random.choice(
            ["VoIP", "Video", "BestEffort"],
            size=n_users,
            p=[0.2, 0.3, 0.5],
        )

        # SINR distribution (typical indoor)
        sinr = np.random.normal(15, 8, n_users)
        sinr = np.clip(sinr, -5, 40)

        # Throughput (correlated with SINR)
        sinr_linear = 10 ** (sinr / 10)
        throughput = 20 * np.log2(1 + sinr_linear) / n_users * n_rus
        throughput = np.clip(throughput, 0.1, 100)

        # RU association (nearest RU)
        ru_positions = np.column_stack([
            np.linspace(5, arena_size[0] - 5, int(np.ceil(np.sqrt(n_rus)))).repeat(
                int(np.ceil(np.sqrt(n_rus)))
            )[:n_rus],
            np.tile(
                np.linspace(5, arena_size[1] - 5, int(np.ceil(np.sqrt(n_rus)))),
                int(np.ceil(np.sqrt(n_rus))),
            )[:n_rus],
        ])
        ru_assoc = np.zeros(n_users, dtype=int)
        for u in range(n_users):
            dists = np.linalg.norm(ru_positions - positions[u], axis=1)
            ru_assoc[u] = dists.argmin()

        return {
            "user_positions": positions,
            "traffic_types": traffic_types,
            "sinr": sinr,
            "throughput": throughput,
            "ru_association": ru_assoc,
            "ru_positions": ru_positions,
            "n_samples": n_users,
            "n_rus": n_rus,
        }

    def get_qos_distributions(self, data: dict) -> dict:
        """Extract QoS class distributions for environment parameterization."""
        if data.get("traffic_types") is not None:
            types = data["traffic_types"]
            if isinstance(types[0], str):
                return {
                    "VoIP": (types == "VoIP").mean(),
                    "Video": (types == "Video").mean(),
                    "BestEffort": (types == "BestEffort").mean(),
                }
            else:
                unique, counts = np.unique(types, return_counts=True)
                return dict(zip(unique, counts / counts.sum()))
        return {"VoIP": 0.2, "Video": 0.3, "BestEffort": 0.5}

    def get_sinr_statistics(self, data: dict) -> dict:
        """Get SINR distribution parameters for realistic simulation."""
        if data.get("sinr") is not None:
            return {
                "mean": float(np.mean(data["sinr"])),
                "std": float(np.std(data["sinr"])),
                "p5": float(np.percentile(data["sinr"], 5)),
                "p95": float(np.percentile(data["sinr"], 95)),
            }
        return {"mean": 15.0, "std": 8.0, "p5": 2.0, "p95": 30.0}
