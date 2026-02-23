# STAG-FedSAC: Spatio-Temporal Attention Graph with Federated Soft Actor-Critic for Transit-Venue WiFi Optimization

> **Target Journal**: Wireless Personal Communications (Springer)

## Overview

STAG-FedSAC is a novel algorithm for proactive WiFi resource management in high-density transit venues (airports, metro stations, bus terminals). It jointly solves spatio-temporal load prediction, proactive resource pre-configuration, and fair multi-QoS bandwidth allocation in a privacy-preserving, distributed fashion.

## Novel Contributions

1. **ST-GCAT** — Schedule-Conditioned Graph Cross-Attention Transformer: The first graph-attention + Transformer predictor for WiFi AP load that uses structured transit schedule data as cross-attention features.

2. **Graph-Embedded SAC State** — The first use of graph prediction embeddings (dense, spatially-structured tensors) as structured DRL state input for wireless resource allocation.

3. **Bidirectional Joint Training** — The first algorithm where policy gradient flows back through the predictor during joint end-to-end training (L_pred = MSE + η·L_actor).

4. **HierFed-KD** — Hierarchical Federated Aggregation with Knowledge Distillation: Two-level zone/global federation with 10× communication compression.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      STAG-FedSAC System                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Module 1: ST-GCAT Predictor                                   │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────────────┐      │
│  │Spatial   │→│Temporal      │→│Schedule Cross-Attn  │      │
│  │GAT Layer │  │Transformer   │  │(Core Novelty)       │      │
│  └──────────┘  └──────────────┘  └─────────┬───────────┘      │
│                                             │                  │
│                              Z_fused (graph embeddings)        │
│                                             │                  │
│  Module 2: Graph-SAC Agent                  ↓                  │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ SAC Actor  │ Double-Q Critic │ Hybrid Actions        │      │
│  │ (per-AP)   │ (per-AP)        │ Power+Channel+BW      │      │
│  └──────────────────────────────────────────────────────┘      │
│                              │                                 │
│  Module 3: HierFed-KD        ↓                                 │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ Zone FedAvg │ Global FedProx │ Knowledge Distillation│      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                │
│  >>>>>>> Bidirectional Gradient Flow (η·L_actor → φ) <<<<<<<  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
stag_fedsac/
├── models/
│   ├── stgcat.py             # ST-GCAT: SpatialGAT, TemporalTransformer,
│   │                         #          ScheduleCrossAttention, PredictionHead
│   ├── graph_sac.py          # SAC Actor, Critic, PersonalizedSACActor
│   └── hierfed_kd.py         # Zone aggregation, FedProx, KD distillation
├── training/
│   ├── joint_trainer.py      # Main training loop (Algorithm steps 1-16)
│   ├── replay_buffer.py      # Per-AP experience replay
│   └── lagrangian.py         # Adaptive Lagrange multiplier updates
├── environment/
│   ├── wifi_env.py           # Python WiFi simulation environment
│   ├── schedule_generator.py # Transit schedule feature tensor generator
│   └── graph_builder.py      # Interference graph construction
├── data/
│   ├── dartmouth_loader.py   # Dartmouth'18 preprocessing pipeline
│   ├── lcr_hdd_loader.py     # LCR HDD ACC Arena preprocessing
│   └── waca_loader.py        # WACA Camp Nou interference matrix
├── evaluation/
│   ├── metrics.py            # All evaluation metrics (Section 9)
│   ├── baselines.py          # SSF, LLF, LSTM-DRL, FedDDPG
│   └── ablation.py           # Automated ablation experiment runner
└── config.py                 # All hyperparameters
```

## Datasets

| Dataset | DOI | Use |
|---------|-----|-----|
| Dartmouth'18 Campus WiFi | 10.15783/C7F59T | ST-GCAT training (AP load time series) |
| LCR HDD 5G High Density | 10.1038/s41597-025-06282-0 | DRL training & QoS evaluation |
| WACA Camp Nou | 10.5281/zenodo.3960029 | Channel interference calibration |

## Setup

```bash
# Create conda environment (Python 3.13 with CUDA)
conda create -n py313 python=3.13 -y
conda activate py313

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Download datasets
python download_data.py
```

## Usage

### Training

```bash
# Full STAG-FedSAC training
python main.py train --episodes 2000 --n-aps 30

# Quick training (reduced scale)
python main.py train --episodes 100 --n-aps 10

# Ablation: no joint training
python main.py train --no-joint --episodes 100
```

### Evaluation

```bash
# Evaluate all methods (proposed + baselines)
python main.py evaluate --eval-episodes 20

# Run full ablation study
python main.py ablation --episodes 100 --n-aps 10
```

### TensorBoard

```bash
tensorboard --logdir logs/
```

## Baselines

| # | Method | Prediction | Policy | Federated | Schedule | Joint Training |
|---|--------|-----------|--------|-----------|----------|---------------|
| 1 | SSF | None | Rule | No | No | N/A |
| 2 | LLF | None | Rule | No | No | N/A |
| 3 | LSTM-DRL | LSTM (scalar) | DDPG | No | No | No |
| 4 | FedDDPG | None | DDPG | Yes | No | N/A |
| 5 | No Joint | ST-GCAT | SAC | Yes | Yes | **No** |
| 6 | No Schedule | ST-GCAT (no S) | SAC | Yes | **No** | Yes |
| 7 | **STAG-FedSAC** | **ST-GCAT** | **SAC** | **Yes (2-level+KD)** | **Yes** | **Yes** |

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| η (eta_joint) | 0.1 | Policy gradient weight in predictor loss |
| α (throughput) | 0.40 | Throughput reward weight |
| β (latency) | 0.25 | Latency reward weight |
| γ (fairness) | 0.20 | Fairness reward weight |
| δ (prediction) | 0.10 | Prediction bonus weight |
| T_local | 100 | Zone aggregation interval |
| T_global | 500 | Global aggregation interval |

## Citation

If you use this code, please cite:

```bibtex
@article{stag_fedsac_2026,
  title={STAG-FedSAC: Spatio-Temporal Attention Graph with Federated Soft
         Actor-Critic for Transit-Venue WiFi Optimization},
  journal={Wireless Personal Communications},
  year={2026}
}
```

## License

MIT License
