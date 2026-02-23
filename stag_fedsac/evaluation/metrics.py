"""
Evaluation Metrics — All metrics from Section 9.
"""
import numpy as np
import torch


def compute_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Root Mean Square Error for load predictions."""
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean Absolute Error for load predictions."""
    return float(np.mean(np.abs(predictions - targets)))


def compute_mape(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    mask = targets > 0.01  # avoid division by zero
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(predictions[mask] - targets[mask]) / targets[mask]) * 100)


def compute_surge_rmse(
    predictions: np.ndarray,
    targets: np.ndarray,
    surge_mask: np.ndarray,
) -> float:
    """RMSE during schedule-driven surge windows only.

    Novel metric: prediction accuracy during the hardest windows.
    """
    if surge_mask.sum() == 0:
        return 0.0
    return float(np.sqrt(np.mean((predictions[surge_mask] - targets[surge_mask]) ** 2)))


def compute_jain_fairness(rates: np.ndarray) -> float:
    """Jain's Fairness Index: (Σr)² / (n · Σr²)."""
    if len(rates) == 0 or np.all(rates == 0):
        return 1.0
    n = len(rates)
    return float((rates.sum() ** 2) / (n * (rates ** 2).sum()))


def compute_max_min_fairness(rates: np.ndarray) -> float:
    """Min/Max ratio fairness metric."""
    if len(rates) == 0 or rates.max() == 0:
        return 1.0
    return float(rates.min() / rates.max())


def compute_qos_satisfaction(
    sinr_values: np.ndarray,
    qos_classes: np.ndarray,
    thresholds: dict[int, float] = None,
) -> dict:
    """QoS satisfaction rate per class and overall.

    Args:
        sinr_values: [N_users] SINR in dB
        qos_classes: [N_users] class indices (0=VoIP, 1=Video, 2=BE)
        thresholds: class → SINR threshold mapping
    """
    if thresholds is None:
        thresholds = {0: 10.0, 1: 15.0, 2: 5.0}

    result = {"overall": 0.0}
    total_satisfied = 0
    total_users = len(sinr_values)

    for cls, threshold in thresholds.items():
        mask = qos_classes == cls
        n_class = mask.sum()
        if n_class == 0:
            result[f"class_{cls}"] = 1.0
            continue
        satisfied = (sinr_values[mask] >= threshold).sum()
        result[f"class_{cls}"] = float(satisfied / n_class)
        total_satisfied += satisfied

    result["overall"] = float(total_satisfied / max(total_users, 1))
    return result


def compute_system_throughput(throughputs: np.ndarray, n_users: int) -> float:
    """Average system throughput per user (Mbps/user)."""
    if n_users == 0:
        return 0.0
    return float(throughputs.sum() / n_users)


def compute_energy_efficiency(throughput: float, powers: np.ndarray) -> float:
    """Energy efficiency: throughput / total power (Mbps/W)."""
    total_power_w = np.sum(10 ** (powers / 10) / 1000)  # convert dBm to W
    if total_power_w == 0:
        return 0.0
    return float(throughput / total_power_w)


def compute_handover_rate(
    associations: list[np.ndarray], n_users: int, n_steps: int
) -> float:
    """Handover rate: association changes per user per hour."""
    if n_users == 0 or n_steps <= 1:
        return 0.0
    changes = 0
    for t in range(1, len(associations)):
        changes += (associations[t] != associations[t - 1]).sum()
    return float(changes / n_users)


def compute_communication_overhead(
    kd_size: int, raw_param_size: int
) -> float:
    """Communication overhead ratio: KD message size / raw parameter size."""
    if raw_param_size == 0:
        return 0.0
    return float(kd_size / raw_param_size)


def compute_fairness_efficiency_product(jfi: float, throughput: float) -> float:
    """Combined fairness-efficiency metric: JFI × Throughput."""
    return float(jfi * throughput)


def compute_all_metrics(
    predictions: np.ndarray | None = None,
    targets: np.ndarray | None = None,
    throughputs: np.ndarray | None = None,
    latencies: np.ndarray | None = None,
    sinr_values: np.ndarray | None = None,
    qos_classes: np.ndarray | None = None,
    loads: np.ndarray | None = None,
    powers: np.ndarray | None = None,
    surge_mask: np.ndarray | None = None,
    n_users: int = 0,
) -> dict:
    """Compute all evaluation metrics."""
    metrics = {}

    # Prediction quality (Section 9.1)
    if predictions is not None and targets is not None:
        metrics["rmse"] = compute_rmse(predictions, targets)
        metrics["mae"] = compute_mae(predictions, targets)
        metrics["mape"] = compute_mape(predictions, targets)
        if surge_mask is not None:
            metrics["surge_rmse"] = compute_surge_rmse(predictions, targets, surge_mask)

    # System performance (Section 9.2)
    if throughputs is not None:
        metrics["system_throughput"] = compute_system_throughput(throughputs, n_users)
    if latencies is not None:
        metrics["avg_latency"] = float(np.mean(latencies))
        metrics["p95_latency"] = float(np.percentile(latencies, 95)) if len(latencies) > 0 else 0.0
    if loads is not None:
        metrics["load_std"] = float(np.std(loads))
    if throughputs is not None and powers is not None:
        metrics["energy_efficiency"] = compute_energy_efficiency(
            throughputs.sum(), powers
        )

    # Fairness (Section 9.3)
    if throughputs is not None:
        metrics["jain_fairness"] = compute_jain_fairness(throughputs)
        metrics["max_min_fairness"] = compute_max_min_fairness(throughputs)
        metrics["fairness_efficiency"] = compute_fairness_efficiency_product(
            metrics["jain_fairness"], metrics.get("system_throughput", 0)
        )
    if sinr_values is not None and qos_classes is not None:
        qos = compute_qos_satisfaction(sinr_values, qos_classes)
        metrics.update({f"qos_{k}": v for k, v in qos.items()})

    return metrics
