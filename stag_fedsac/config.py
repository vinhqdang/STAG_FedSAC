"""
STAG-FedSAC Configuration — All hyperparameters from Section 11.
"""
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────── Network / Environment ────────────────
N_AP = 30                  # access points
N_USERS_MAX = 1000         # max simultaneous users
N_CHANNELS = 11            # 2.4 GHz channels
FREQUENCY_GHZ = 2.4
CHANNEL_WIDTH_MHZ = 20
P_MIN = 5.0                # dBm
P_MAX = 23.0               # dBm
L_MAX = 0.90               # max load fraction
SIM_DURATION_S = 3600      # 1 hour per episode
TIMESTEP_S = 300           # 5-minute control interval
STEPS_PER_EPISODE = SIM_DURATION_S // TIMESTEP_S  # 12 steps

# ──────────────── ST-GCAT Predictor ────────────────
T_HISTORY = 48             # historical window (4 hours @ 5 min)
F_FEATURES = 5             # load, throughput, latency, channel_util, user_count
DELTA_HORIZON = 6          # prediction horizon (30 min ahead)
D_SCHEDULE = 8             # schedule feature dim
D_HIDDEN = 128             # hidden dimension
N_HEADS = 4                # attention heads
N_TRANSFORMER_LAYERS = 2
DROPOUT = 0.1

# ──────────────── SAC Agent ────────────────
STATE_DIM = F_FEATURES + DELTA_HORIZON * D_HIDDEN + 3 + D_SCHEDULE  # 784
ACTION_DIM = 1 + N_CHANNELS + 3     # power + channel logits + bw fractions
HIDDEN_DIM_SAC = 256
GAMMA_DISCOUNT = 0.99
TAU_SOFT = 0.005
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
LR_PREDICTOR = 1e-3
LR_FEDERATED_KD = 5e-4

# ──────────────── Reward Weights ────────────────
ALPHA_REWARD = 0.40        # throughput
BETA_REWARD = 0.25         # latency
GAMMA_REWARD = 0.20        # fairness
DELTA_REWARD = 0.10        # prediction bonus
MU_REWARD = 0.05           # QoS violation
RHO_LAGRANGIAN = 0.01      # Lagrangian step size

# ──────────────── Joint Training ────────────────
ETA_JOINT = 0.1            # policy gradient weight in L_pred
BATCH_SIZE = 256
REPLAY_BUFFER_SIZE = 100_000
UPDATE_FREQ = 4            # update every 4 steps
WARMUP_STEPS = 10_000
TOTAL_EPISODES = 2000
MAX_STEPS_PER_EP = 720

# ──────────────── Federation ────────────────
T_LOCAL_FED = 100           # zone aggregation interval
T_GLOBAL_FED = 500          # global aggregation interval
FEDPROX_MU = 0.01           # proximal term weight
S_REF_SIZE = 100            # reference states for KD
N_ZONES = 3                 # number of venue zones

# ──────────────── QoS Classes ────────────────
QOS_CLASSES = {
    "VoIP": {"sinr_threshold": 10.0, "latency_sla_ms": 50},
    "Video": {"sinr_threshold": 15.0, "latency_sla_ms": 100},
    "BestEffort": {"sinr_threshold": 5.0, "latency_sla_ms": 500},
}

# ──────────────── Data Paths ────────────────
DATA_DIR = "data"
DARTMOUTH_DIR = f"{DATA_DIR}/dartmouth"
LCR_HDD_DIR = f"{DATA_DIR}/lcr_hdd"
WACA_DIR = f"{DATA_DIR}/waca"
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
