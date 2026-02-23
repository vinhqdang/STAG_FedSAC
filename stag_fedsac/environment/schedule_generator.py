"""
Schedule Generator — Transit schedule feature tensor generation.
"""
import numpy as np


class ScheduleGenerator:
    """Generates transit schedule features for ST-GCAT.

    Creates Gaussian surge functions centered at event times,
    producing D_s=8 features per AP per future timestep.
    """

    def __init__(
        self,
        n_aps: int,
        delta: int = 6,
        d_schedule: int = 8,
        sigma: float = 2.0,  # Gaussian width in timesteps (~10 min)
        zone_assignments: dict[int, int] | None = None,
    ):
        self.n_aps = n_aps
        self.delta = delta
        self.d_schedule = d_schedule
        self.sigma = sigma
        self.zone_assignments = zone_assignments or {i: i % 3 for i in range(n_aps)}

    def generate_schedule_tensor(
        self,
        current_step: int,
        events: list[dict],
    ) -> np.ndarray:
        """Generate schedule feature tensor for all APs and future timesteps.

        Args:
            current_step: current simulation step
            events: list of schedule events with zone, time_offset, n_passengers, type

        Returns:
            S: [N_ap, delta, D_s] schedule feature tensor
        """
        S = np.zeros((self.n_aps, self.delta, self.d_schedule))

        for dt in range(self.delta):
            future_step = current_step + dt + 1

            for event in events:
                tau = event["time_offset"] - future_step
                surge = np.exp(-tau ** 2 / (2 * self.sigma ** 2))
                zone = event["zone"]

                for ap_id in range(self.n_aps):
                    if self.zone_assignments[ap_id] == zone:
                        if event["type"] == "departure":
                            S[ap_id, dt, 0] += event["n_passengers"] * surge
                            S[ap_id, dt, 3] = tau if tau > 0 else 0
                        else:
                            S[ap_id, dt, 1] += event["n_passengers"] * surge
                            S[ap_id, dt, 4] = -tau if tau < 0 else 0

                        S[ap_id, dt, 2] = 1.0 if abs(tau) < 3 else 0.0
                        # Zone type one-hot
                        z_idx = min(zone, 2)
                        S[ap_id, dt, 5 + z_idx] = 1.0
                        # Occupancy forecast
                        S[ap_id, dt, 7] = min(
                            1.0,
                            S[ap_id, dt, 0] / 300 + S[ap_id, dt, 1] / 300,
                        )

        return S.astype(np.float32)

    def generate_random_events(
        self, episode_len: int, n_events: int = 5
    ) -> list[dict]:
        """Generate random schedule events for an episode."""
        events = []
        for _ in range(n_events):
            events.append({
                "zone": np.random.randint(0, 3),
                "time_offset": np.random.uniform(0, episode_len),
                "n_passengers": np.random.randint(50, 300),
                "type": np.random.choice(["departure", "arrival"]),
            })
        return events

    def generate_from_class_schedule(
        self, class_times: list[float], zone_map: dict[float, int]
    ) -> list[dict]:
        """Domain adaptation: campus class times → transit departures.

        Used for Dartmouth dataset adaptation.
        """
        events = []
        for t in class_times:
            zone = zone_map.get(t, 0)
            events.append({
                "zone": zone,
                "time_offset": t,
                "n_passengers": np.random.randint(30, 200),
                "type": "departure",
            })
            # Class end → arrival
            events.append({
                "zone": zone,
                "time_offset": t + 10,  # 50 minutes later
                "n_passengers": np.random.randint(30, 200),
                "type": "arrival",
            })
        return events
