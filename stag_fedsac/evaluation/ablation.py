"""
Ablation Study — Automated ablation experiment runner.
"""
import json
import os
from collections import defaultdict

import numpy as np

from stag_fedsac.training.joint_trainer import JointTrainer
from stag_fedsac.evaluation.baselines import (
    SSFBaseline, LLFBaseline, LSTMDRLBaseline, FedDDPGBaseline,
    evaluate_baseline,
)
from stag_fedsac.environment.wifi_env import WiFiEnvironment
from stag_fedsac.config import *


ABLATION_CONFIGS = {
    "full": {
        "description": "STAG-FedSAC (proposed) — full system",
        "use_joint_training": True,
        "use_schedule": True,
        "use_hierarchical_fed": True,
        "use_embedding": True,
    },
    "no_joint": {
        "description": "A1: η=0 (no joint gradient) — validates Contribution 3",
        "use_joint_training": False,
        "use_schedule": True,
        "use_hierarchical_fed": True,
        "use_embedding": True,
    },
    "no_schedule": {
        "description": "A2: No schedule input S — validates Contribution 1",
        "use_joint_training": True,
        "use_schedule": False,
        "use_hierarchical_fed": True,
        "use_embedding": True,
    },
    "no_embedding": {
        "description": "A3: No embedding state S_e — validates Contribution 2",
        "use_joint_training": True,
        "use_schedule": True,
        "use_hierarchical_fed": True,
        "use_embedding": False,
    },
    "no_hier_fed": {
        "description": "A4: Single-level FL — validates Contribution 4",
        "use_joint_training": True,
        "use_schedule": True,
        "use_hierarchical_fed": False,
        "use_embedding": True,
    },
}


def run_ablation_study(
    n_aps: int = 10,
    total_episodes: int = 100,
    eval_episodes: int = 10,
    output_dir: str = "results/ablation",
) -> dict:
    """Run all ablation experiments and baselines.

    Args:
        n_aps: Number of APs (reduced for faster ablation)
        total_episodes: Training episodes per configuration
        eval_episodes: Evaluation episodes
        output_dir: Directory for results

    Returns:
        Dictionary of results per configuration
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    # ── Evaluate baselines ──
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION")
    print("=" * 60)

    env = WiFiEnvironment(n_aps=n_aps, n_channels=N_CHANNELS)

    baselines = {
        "SSF": SSFBaseline(n_aps, N_CHANNELS),
        "LLF": LLFBaseline(n_aps, N_CHANNELS),
        "LSTM-DRL": LSTMDRLBaseline(n_aps, N_CHANNELS),
        "FedDDPG": FedDDPGBaseline(n_aps, N_CHANNELS),
    }

    for name, baseline in baselines.items():
        print(f"\nEvaluating {name}...")
        result = evaluate_baseline(baseline, env, eval_episodes, name)
        all_results[name] = result
        print(f"  {name}: {result}")

    # ── Run ablation configurations ──
    print("\n" + "=" * 60)
    print("ABLATION EXPERIMENTS")
    print("=" * 60)

    for config_name, config in ABLATION_CONFIGS.items():
        print(f"\n{'─' * 40}")
        print(f"Configuration: {config_name}")
        print(f"Description: {config['description']}")
        print(f"{'─' * 40}")

        trainer = JointTrainer(
            n_aps=n_aps,
            n_channels=N_CHANNELS,
            log_dir=f"{output_dir}/logs_{config_name}",
            checkpoint_dir=f"{output_dir}/ckpt_{config_name}",
            use_joint_training=config["use_joint_training"],
            use_schedule=config["use_schedule"],
            use_hierarchical_fed=config["use_hierarchical_fed"],
            use_embedding=config.get("use_graph_embedding", True),  # A3 ablation
        )

        history = trainer.train(
            total_episodes=total_episodes,
            eval_interval=max(1, total_episodes // 5),
        )

        # Final evaluation
        eval_metrics = trainer.evaluate(n_episodes=eval_episodes)
        eval_metrics["name"] = config_name
        eval_metrics["description"] = config["description"]
        all_results[config_name] = eval_metrics
        print(f"\nFinal metrics: {eval_metrics}")

    # ── Save results ──
    results_path = os.path.join(output_dir, "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ── Print summary table ──
    print_results_table(all_results)

    return all_results


def print_results_table(results: dict) -> None:
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 100)

    headers = ["Method", "Reward", "Throughput", "Latency", "JFI", "Load Std"]
    format_str = "{:<25} {:<12} {:<12} {:<12} {:<12} {:<12}"
    print(format_str.format(*headers))
    print("-" * 100)

    for name, metrics in results.items():
        row = [
            name,
            f"{metrics.get('reward', 0):.4f}",
            f"{metrics.get('throughput', 0):.4f}",
            f"{metrics.get('latency', 0):.2f}",
            f"{metrics.get('jain_fairness', 0):.4f}",
            f"{metrics.get('load_std', 0):.4f}",
        ]
        print(format_str.format(*row))

    print("=" * 100)


if __name__ == "__main__":
    run_ablation_study(
        n_aps=10,
        total_episodes=50,
        eval_episodes=5,
    )
