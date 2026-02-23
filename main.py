"""
STAG-FedSAC — Main entry point for training and evaluation.
"""
import argparse
import os
import json
import sys

import torch

from stag_fedsac.config import *
from stag_fedsac.training.joint_trainer import JointTrainer
from stag_fedsac.evaluation.ablation import run_ablation_study
from stag_fedsac.evaluation.baselines import (
    SSFBaseline, LLFBaseline, LSTMDRLBaseline, FedDDPGBaseline,
    evaluate_baseline,
)
from stag_fedsac.environment.wifi_env import WiFiEnvironment


def train(args):
    """Train STAG-FedSAC with full configuration."""
    print(f"Device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    trainer = JointTrainer(
        n_aps=args.n_aps,
        n_channels=args.n_channels,
        device=DEVICE,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        use_joint_training=not args.no_joint,
        use_schedule=not args.no_schedule,
        use_hierarchical_fed=not args.no_hier_fed,
    )

    history = trainer.train(
        total_episodes=args.episodes,
        eval_interval=args.eval_interval,
    )

    # Save training history
    os.makedirs("results", exist_ok=True)
    with open("results/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete. Results saved to results/")


def evaluate(args):
    """Evaluate all methods (proposed + baselines)."""
    print("Running full evaluation...")

    env = WiFiEnvironment(
        n_aps=args.n_aps,
        n_channels=args.n_channels,
    )

    results = {}

    # Baselines
    baselines = {
        "SSF": SSFBaseline(args.n_aps, args.n_channels),
        "LLF": LLFBaseline(args.n_aps, args.n_channels),
        "LSTM-DRL": LSTMDRLBaseline(args.n_aps, args.n_channels),
        "FedDDPG": FedDDPGBaseline(args.n_aps, args.n_channels),
    }

    for name, baseline in baselines.items():
        print(f"\nEvaluating {name}...")
        result = evaluate_baseline(baseline, env, args.eval_episodes, name)
        results[name] = result
        print(f"  {result}")

    # STAG-FedSAC (if checkpoint exists)
    ckpt_path = os.path.join(args.checkpoint_dir, "checkpoint_best.pt")
    if os.path.exists(ckpt_path):
        print("\nEvaluating STAG-FedSAC (trained)...")
        trainer = JointTrainer(
            n_aps=args.n_aps,
            n_channels=args.n_channels,
        )
        trainer.load_checkpoint("best")
        eval_result = trainer.evaluate(n_episodes=args.eval_episodes)
        eval_result["name"] = "STAG-FedSAC"
        results["STAG-FedSAC"] = eval_result
    else:
        print(f"\nNo checkpoint found at {ckpt_path}. Run training first.")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print table
    from stag_fedsac.evaluation.ablation import print_results_table
    print_results_table(results)


def ablation(args):
    """Run ablation study."""
    run_ablation_study(
        n_aps=args.n_aps,
        total_episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        output_dir="results/ablation",
    )


def main():
    parser = argparse.ArgumentParser(
        description="STAG-FedSAC: Spatio-Temporal Attention Graph with "
        "Federated Soft Actor-Critic for Transit-Venue WiFi Optimization"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train
    train_parser = subparsers.add_parser("train", help="Train STAG-FedSAC")
    train_parser.add_argument("--n-aps", type=int, default=N_AP)
    train_parser.add_argument("--n-channels", type=int, default=N_CHANNELS)
    train_parser.add_argument("--episodes", type=int, default=TOTAL_EPISODES)
    train_parser.add_argument("--eval-interval", type=int, default=50)
    train_parser.add_argument("--log-dir", type=str, default=LOG_DIR)
    train_parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR)
    train_parser.add_argument("--no-joint", action="store_true",
                              help="Disable joint training (ablation A1)")
    train_parser.add_argument("--no-schedule", action="store_true",
                              help="Disable schedule features (ablation A2)")
    train_parser.add_argument("--no-hier-fed", action="store_true",
                              help="Disable hierarchical federation (ablation A4)")

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate all methods")
    eval_parser.add_argument("--n-aps", type=int, default=N_AP)
    eval_parser.add_argument("--n-channels", type=int, default=N_CHANNELS)
    eval_parser.add_argument("--eval-episodes", type=int, default=20)
    eval_parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR)

    # Ablation
    abl_parser = subparsers.add_parser("ablation", help="Run ablation study")
    abl_parser.add_argument("--n-aps", type=int, default=10)
    abl_parser.add_argument("--episodes", type=int, default=100)
    abl_parser.add_argument("--eval-episodes", type=int, default=10)

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "ablation":
        ablation(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
