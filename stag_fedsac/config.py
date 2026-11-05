"""
STAG-FedSAC Configuration — All hyperparameters from Section 11.
"""
import os

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────── Network / Environment ────────────────
N_AP = 30                  # access points
N_USERS_MAX = 1000         # max simultaneous users
N_CHANNELS = 11            # orthogonal 20 MHz channels (5 GHz)
FREQUENCY_GHZ = 5.0
CHANNEL_WIDTH_MHZ = 20
P_MIN = 5.0                # dBm
P_MAX = 23.0               # dBm
L_MAX = 0.90               # max load fraction
SIM_DURATION_S = 3600      # 1 hour per episode
TIMESTEP_S = 300           # 5-minute control interval
STEPS_PER_EPISODE = SIM_DURATION_S // TIMESTEP_S  # 12 steps

# ──────────────── ST-GCAT Predictor ────────────────
T_HISTORY = 12             # historical window (60 min @ 5 min)
F_FEATURES = 5             # load, throughput, latency, channel_util, user_count
DELTA_HORIZON = 6          # prediction horizon (30 min ahead)
D_SCHEDULE = 8             # schedule feature dim
D_HIDDEN = 128             # hidden dimension
N_HEADS = 4                # attention heads
N_TRANSFORMER_LAYERS = 2
DROPOUT = 0.1

# ──────────────── SAC Agent ────────────────
RADIO_DIM = N_CHANNELS + 2   # own channel (one-hot), users and APs sensed in its collision domain
STATE_DIM = F_FEATURES + DELTA_HORIZON * D_HIDDEN + 3 + D_SCHEDULE + RADIO_DIM  # 797
ACTION_DIM = 1 + N_CHANNELS + 3     # power + channel logits + bw fractions
HIDDEN_DIM_SAC = 256
GAMMA_DISCOUNT = 0.99
TAU_SOFT = 0.005
LR_ACTOR = float(os.environ.get("STAG_LR_ACTOR", 3e-4))
LR_CRITIC = 3e-4
LR_PREDICTOR = 5e-4
LR_FEDERATED_KD = 5e-4

# ──────────────── Reward Weights ────────────────
ALPHA_REWARD = 0.40        # throughput
BETA_REWARD = 0.25         # latency
GAMMA_REWARD = 0.20        # fairness
DELTA_REWARD = 0.10        # prediction bonus
MU_REWARD = 0.05           # QoS violation (weight "delta" of Eq. 1 in the manuscript)
RHO_LAGRANGIAN = 0.01      # Lagrangian step size

# ──────────────── Joint Training ────────────────
ETA_JOINT = 0.1            # policy gradient weight in L_pred
BATCH_SIZE = 256
REPLAY_BUFFER_SIZE = 100_000
UPDATE_FREQ = int(os.environ.get("STAG_UPDATE_FREQ", 1))   # gradient update every k env steps (update-to-data ratio 1)
WARMUP_STEPS = 2_400        # 200 episodes x 12 steps of random exploration
TOTAL_EPISODES = 2000
MAX_STEPS_PER_EP = 12
PRED_BATCH = 16            # (history, schedule) samples per predictor update
PRED_BUFFER = 3000
GUMBEL_TAU_START = 2.0
GUMBEL_TAU_END = 0.5
SWITCH_OUTAGE_S = 3.0      # service outage caused by one channel switch (CSA + re-association)
EDCA_LEVELS = 16           # airtime budget granularity per access category
UTIL_LIMIT = 0.90          # airtime-utilisation limit enforced by the Lagrangian
LAMBDA_MAX = 1.0           # upper bound of the Lagrange multiplier (keeps the penalty on the reward scale)
UTIL_VIOL_CLIP = 0.3       # violations beyond this are not penalised further

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

# ──────────────── SAC scale parameters (shared by all learned methods) ────────────────
R_SCALE = float(os.environ.get("STAG_R_SCALE", 10.0))      # reward scaling (per-step reward is O(0.1))
ALPHA_INIT = float(os.environ.get("STAG_ALPHA_INIT", 0.05))  # initial SAC entropy temperature
TARGET_ENTROPY = float(os.environ.get("STAG_TARGET_ENTROPY", -3.0))
BC_WEIGHT = float(os.environ.get("STAG_BC", 0.0))      # weight of the incumbent-action prior in the actor loss
DEFAULT_POWER_DBM = 14.0                               # incumbent configuration: mid power, round-robin channels, equal airtime
EMB_SCALE = float(os.environ.get("STAG_EMB_SCALE", 0.03))      # scale of the embedding block in the state (norm matched to the other features)
