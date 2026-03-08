"""
Quick Smoke Test for STAG-FedSAC.

Trains for a handful of episodes with a small configuration to verify
the full training loop runs without errors.

Usage:
    conda activate py313
    python run_quick_test.py
"""
import sys
import time
import torch

print("=" * 60)
print("STAG-FedSAC Quick Smoke Test")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

from stag_fedsac.training.joint_trainer import JointTrainer
from stag_fedsac.evaluation.baselines import (
    SSFBaseline, LLFBaseline, evaluate_baseline,
)
from stag_fedsac.environment.wifi_env import WiFiEnvironment
from stag_fedsac.config import DEVICE, N_CHANNELS

# ── Config for smoke test ──
N_APS = 5
EPISODES = 5
EVAL_EPISODES = 2

print(f"Config: N_APs={N_APS}, episodes={EPISODES}, device={DEVICE}")
print("-" * 60)

# ── 1. Test baselines ──
print("\n[1/3] Testing baselines...")
env = WiFiEnvironment(n_aps=N_APS, n_channels=N_CHANNELS)
for name, baseline in [("SSF", SSFBaseline(N_APS, N_CHANNELS)),
                        ("LLF", LLFBaseline(N_APS, N_CHANNELS))]:
    result = evaluate_baseline(baseline, env, n_episodes=2, baseline_name=name)
    print(f"  {name}: reward={result['reward']:.4f}, throughput={result['throughput']:.4f}")

# ── 2. Test full STAG-FedSAC training ──
print(f"\n[2/3] Training STAG-FedSAC ({EPISODES} episodes)...")
t0 = time.time()
trainer = JointTrainer(
    n_aps=N_APS,
    n_channels=N_CHANNELS,
    device=DEVICE,
    log_dir="logs/smoke_test",
    checkpoint_dir="checkpoints/smoke_test",
    use_joint_training=True,
    use_schedule=True,
    use_hierarchical_fed=True,
)

history = trainer.train(
    total_episodes=EPISODES,
    eval_interval=EPISODES + 1,  # no mid-training eval for speed
)
elapsed = time.time() - t0
print(f"  Training complete in {elapsed:.1f}s")
print(f"  Final reward: {history['episode_reward'][-1]:.4f}")
print(f"  Avg reward: {sum(history['episode_reward']) / len(history['episode_reward']):.4f}")

# ── 3. Test evaluation ──
print(f"\n[3/3] Evaluating ({EVAL_EPISODES} episodes)...")
metrics = trainer.evaluate(n_episodes=EVAL_EPISODES)
print("  Evaluation metrics:")
for k, v in metrics.items():
    print(f"    {k}: {v:.4f}")

# ── Summary ──
print("\n" + "=" * 60)
print("✓ SMOKE TEST PASSED — Full training loop functional")
print(f"  Total time: {elapsed:.1f}s")
print(f"  Device: {DEVICE}")
print("=" * 60)
