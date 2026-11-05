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

    # Feature layout (D_s = 8); divide by NORM to obtain O(1) inputs.
    #   0 departure surge   : sum_e n_pax * exp(-tau^2 / 2 sigma^2) over departure events in the AP zone
    #   1 arrival surge     : same for arrival events
    #   2 event window flag : 1 if an event of the zone lies within +-3 steps
    #   3 time to departure : steps until the next departure event of the zone (0 if none)
    #   4 time since arrival: steps since the last arrival event of the zone (0 if none)
    #   5 occupancy forecast: min(1, (f0 + f1) / 300)
    #   6 event count       : number of zone events within the next 6 steps
    #   7 zone id           : zone index / 2
    NORM = np.array([300.0, 300.0, 1.0, 12.0, 12.0, 1.0, 3.0, 1.0], dtype=np.float32)

    def generate_schedule_tensor(self, current_step: int, events: list[dict]) -> np.ndarray:
        """Normalised schedule tensor S [N_ap, delta, D_s] for the next ``delta`` steps."""
        zv = np.array([self.zone_assignments[i] for i in range(self.n_aps)])
        S = np.zeros((self.n_aps, self.delta, self.d_schedule), dtype=np.float32)
        S[:, :, 7] = (zv / 2.0)[:, None]
        for dt in range(self.delta):
            fut = current_step + dt + 1
            for ev in events:
                mask = zv == ev["zone"]
                tau = ev["time_offset"] - fut
                surge = np.exp(-tau ** 2 / (2 * self.sigma ** 2))
                if ev["type"] == "departure":
                    S[mask, dt, 0] += ev["n_passengers"] * surge
                    if tau > 0:
                        cur = S[mask, dt, 3]
                        S[mask, dt, 3] = np.where(cur == 0, tau, np.minimum(cur, tau))
                else:
                    S[mask, dt, 1] += ev["n_passengers"] * surge
                    if tau < 0:
                        S[mask, dt, 4] = np.maximum(S[mask, dt, 4], -tau)
                if abs(tau) < 3:
                    S[mask, dt, 2] = 1.0
                if 0 < ev["time_offset"] - current_step <= self.delta:
                    S[mask, dt, 6] += 1.0
            S[:, dt, 5] = np.minimum(1.0, S[:, dt, 0] / 300 + S[:, dt, 1] / 300)
        return S / self.NORM

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
