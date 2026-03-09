"""
Standalone script to regenerate fig7_architecture.png with
generous margins so nothing is trimmed.
Run: python manuscripts/regen_fig7.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "figures", "fig7_architecture.png")

# ── Canvas ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Shared helpers ───────────────────────────────────────────────────────
def box(x, y, w, h, title, sub="", fc="#E3F2FD", ec="#1565C0", tfs=10, sfs=8.5):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.12",
                          facecolor=fc, edgecolor=ec, linewidth=2.0, zorder=2)
    ax.add_patch(rect)
    ty = y + h/2 + (0.18 if sub else 0)
    ax.text(x + w/2, ty, title, ha="center", va="center",
            fontsize=tfs, fontweight="bold", color=ec, zorder=3)
    if sub:
        ax.text(x + w/2, y + h/2 - 0.28, sub, ha="center", va="center",
                fontsize=sfs, color="#455A64", zorder=3)

def hdr(x, y, w, label, color):
    ax.text(x + w/2, y, label, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=color)

def arr(x1, y1, x2, y2, col="#37474F", lw=1.8, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=col,
                                lw=lw, connectionstyle="arc3,rad=0"))

def label(x, y, text, col="#37474F", fs=8.5, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=col, zorder=4)

# ═══════════════════════════════════════════════════════════════════════
# MODULE BACKGROUNDS
# ═══════════════════════════════════════════════════════════════════════
# M1 background
m1bg = FancyBboxPatch((0.4, 5.5), 12.0, 3.0,
                      boxstyle="round,pad=0.15",
                      facecolor="#F1F8E9", edgecolor="#558B2F",
                      linewidth=1.5, linestyle="--", zorder=1)
ax.add_patch(m1bg)
ax.text(6.4, 8.62, "Module 1: ST-GCAT Predictor  [NOVEL]",
        ha="center", va="center", fontsize=12, fontweight="bold",
        color="#33691E", zorder=3)

# M2 background
m2bg = FancyBboxPatch((0.4, 2.6), 12.0, 2.65,
                      boxstyle="round,pad=0.15",
                      facecolor="#E3F2FD", edgecolor="#1565C0",
                      linewidth=1.5, linestyle="--", zorder=1)
ax.add_patch(m2bg)
ax.text(6.4, 5.35, "Module 2: Graph-SAC Agent  [NOVEL state]",
        ha="center", va="center", fontsize=12, fontweight="bold",
        color="#0D47A1", zorder=3)

# M3 background
m3bg = FancyBboxPatch((0.4, 0.35), 12.0, 2.0,
                      boxstyle="round,pad=0.15",
                      facecolor="#FFF8E1", edgecolor="#E65100",
                      linewidth=1.5, linestyle="--", zorder=1)
ax.add_patch(m3bg)
ax.text(6.4, 2.42, "Module 3: HierFed-KD  [NOVEL federation]",
        ha="center", va="center", fontsize=12, fontweight="bold",
        color="#BF360C", zorder=3)

# ═══════════════════════════════════════════════════════════════════════
# MODULE 1 BLOCKS
# ═══════════════════════════════════════════════════════════════════════
box(0.65, 5.85, 2.5, 2.3,
    "Spatial GAT", "Interference graph A\n(RSSI-based edges)",
    "#DCEDC8", "#33691E", 9.5, 8)
box(3.45, 5.85, 2.7, 2.3,
    "Temporal\nTransformer", "History H[t-T:t]\n(12 steps = 60 min)",
    "#DCEDC8", "#33691E", 9.5, 8)
box(6.45, 5.85, 2.9, 2.3,
    "Schedule\nCross-Attention",
    "S[t:t+Delta]  [NOVEL]\nFlight/train schedule",
    "#C5E1A5", "#1B5E20", 9.5, 8)
box(9.65, 5.85, 2.55, 2.3,
    "Prediction\nHead",
    "L_hat [N x Delta]\nZ_fused -> SAC",
    "#DCEDC8", "#33691E", 9.5, 8)

# M1 arrows
arr(3.15, 7.0, 3.45, 7.0)
arr(6.15, 7.0, 6.45, 7.0)
arr(9.35, 7.0, 9.65, 7.0)

# ═══════════════════════════════════════════════════════════════════════
# MODULE 2 BLOCKS
# ═══════════════════════════════════════════════════════════════════════
box(0.65, 2.85, 3.2, 2.2,
    "SAC Actor (per-AP)",
    "Power | Channel | BW\nGaussian / Gumbel / Dirichlet",
    "#BBDEFB", "#1565C0", 9.5, 8)
box(4.15, 2.85, 3.2, 2.2,
    "Double-Q Critic",
    "Q1, Q2 networks\nEntropy regularised",
    "#BBDEFB", "#1565C0", 9.5, 8)
box(7.65, 2.85, 2.5, 2.2,
    "Replay Buffer",
    "Per-AP ring buffer\n100k transitions",
    "#BBDEFB", "#1565C0", 9.5, 8)
box(10.45, 2.85, 1.8, 2.2,
    "Lagrangian\nMult.",
    "lambda_i\nCapacity C2",
    "#BBDEFB", "#1565C0", 9.5, 8)

# Arrows inside M2
arr(3.85, 3.95, 4.15, 3.95)
arr(7.35, 3.95, 7.65, 3.95)
arr(10.15, 3.95, 10.45, 3.95)

# Z_fused from M1 -> M2 state
arr(10.925, 5.85, 10.925, 5.05, col="#1B5E20", lw=2.2)
label(11.55, 5.45, "Z_fused\n(graph embed)", col="#1B5E20", fs=8.5)

# ═══════════════════════════════════════════════════════════════════════
# BIDIRECTIONAL JOINT GRADIENT BRIDGE  (CORE NOVELTY)
# ═══════════════════════════════════════════════════════════════════════
# Draw as a curved double-headed arrow on the left
ax.annotate("", xy=(1.35, 5.85), xytext=(1.35, 5.05),
            arrowprops=dict(arrowstyle="<|-|>", color="#6A1B9A", lw=2.8,
                            mutation_scale=18))
label(0.05, 5.45,
      "Bidirectional\ngradient bridge\neta * L_actor -> phi\n[NOVEL]",
      col="#6A1B9A", fs=8.2, ha="left")

# ═══════════════════════════════════════════════════════════════════════
# MODULE 3 BLOCKS
# ═══════════════════════════════════════════════════════════════════════
box(0.65, 0.6, 5.4, 1.5,
    "Zone FedAvg  (Level 1)",
    "Quality-weighted aggregation, T_local = 100 steps\nPersonalised base per zone",
    "#FFE0B2", "#E65100", 9.5, 8)
box(6.35, 0.6, 6.0, 1.5,
    "Global FedProx + KD  (Level 2)",
    "FedProx anchor = prev global model, mu=0.01\nKD compression ~11.5x, T_global = 500 steps",
    "#FFE0B2", "#E65100", 9.5, 8)

# M2 -> M3 arrows
arr(2.35, 2.85, 2.35, 2.10, col="#E65100", lw=1.8)
arr(5.75, 2.85, 5.75, 2.10, col="#E65100", lw=1.8)
arr(9.15, 2.85, 9.15, 2.10, col="#E65100", lw=1.8)

# M3 -> M2 feedback
ax.annotate("", xy=(12.8, 2.85), xytext=(12.8, 2.10),
            arrowprops=dict(arrowstyle="<-", color="#E65100", lw=1.8))
label(13.5, 2.45, "Global model\ndistributed\nto APs", col="#E65100", fs=8)

# ═══════════════════════════════════════════════════════════════════════
# ENVIRONMENT BOX (right side)
# ═══════════════════════════════════════════════════════════════════════
box(13.1, 3.7, 2.65, 4.6,
    "Transit\nVenue\nEnv.",
    "30 APs\n3 Zones\nUsers\nSchedule",
    "#FCE4EC", "#880E4F", 9.5, 8)

# Action -> Env
arr(12.25, 3.95, 13.1, 3.95, col="#880E4F", lw=1.8)
label(12.68, 4.25, "Actions\n(p,c,w)", col="#880E4F", fs=8)
# Env -> Obs
arr(13.1, 6.5, 12.2, 6.5, col="#880E4F", lw=1.8)
label(12.65, 6.75, "o, r, L_true", col="#880E4F", fs=8)

# ═══════════════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════════════
legend_patches = [
    mpatches.Patch(facecolor="#DCEDC8", edgecolor="#33691E", label="ST-GCAT (prediction)"),
    mpatches.Patch(facecolor="#BBDEFB", edgecolor="#1565C0", label="Graph-SAC (control)"),
    mpatches.Patch(facecolor="#FFE0B2", edgecolor="#E65100", label="HierFed-KD (federation)"),
    mpatches.Patch(facecolor="white",   edgecolor="#6A1B9A",
                   label="Bidirectional gradient bridge [NOVEL]"),
]
ax.legend(handles=legend_patches, loc="lower right",
          bbox_to_anchor=(1.0, 0.0), fontsize=9,
          framealpha=0.95, edgecolor="#BDBDBD")

plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01)
plt.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.15,
            facecolor="white")
plt.close()
print(f"Saved: {OUT}")
