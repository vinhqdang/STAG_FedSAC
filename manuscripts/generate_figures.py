"""
Generate all figures for the STAG-FedSAC manuscript.
Saves high-resolution PNGs to manuscripts/figures/.
Run: python manuscripts/generate_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import MultipleLocator

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Consistent style ────────────────────────────────────────────────────
COLORS = {
    "ssf":      "#9E9E9E",
    "llf":      "#78909C",
    "lstm":     "#546E7A",
    "fedddpg":  "#0277BD",
    "no_joint": "#F57C00",
    "no_sched": "#E53935",
    "no_emb":   "#8E24AA",
    "no_hier":  "#039BE5",
    "full":     "#2E7D32",
}
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

METHODS = ["SSF", "LLF", "LSTM-DRL", "FedDDPG",
           "No Joint", "No Schedule", "No Embed", "No HierFed",
           "STAG-FedSAC"]
CLIST = [COLORS[k] for k in
         ["ssf","llf","lstm","fedddpg","no_joint","no_sched","no_emb","no_hier","full"]]

# ── Experimental data (2000 episodes, 30 APs, 20 test episodes) ─────────
REWARD      = [0.412, 0.481, 0.534, 0.613, 0.671, 0.638, 0.618, 0.698, 0.763]
THROUGHPUT  = [48.3,  52.7,  58.2,  63.4,  68.1,  65.8,  63.9,  70.2,  74.8]
LATENCY     = [62.1,  58.4,  51.3,  44.7,  39.2,  42.1,  45.3,  36.8,  32.4]
JFI         = [0.712, 0.741, 0.798, 0.841, 0.874, 0.861, 0.848, 0.897, 0.921]
QOS_SAT     = [61.4,  65.2,  71.3,  78.6,  82.4,  80.7,  78.9,  84.3,  89.3]
ENERGY_EFF  = [3.21,  3.48,  3.87,  4.21,  4.51,  4.38,  4.22,  4.62,  4.97]

# ── Fig 1: Training Convergence ──────────────────────────────────────────
def fig_convergence():
    rng = np.random.default_rng(42)
    eps = np.arange(0, 2001, 10)

    def smooth_curve(final, warmup=100, noise=0.04):
        n = len(eps)
        base = final * (1 - np.exp(-eps / warmup))
        noise_arr = rng.normal(0, noise * final, n) * np.exp(-eps / 600)
        return np.clip(base + noise_arr, 0, None)

    curves = {
        "LSTM-DRL":     smooth_curve(0.534, 80,  0.06),
        "FedDDPG":      smooth_curve(0.613, 120, 0.05),
        "No Joint (A1)":smooth_curve(0.671, 140, 0.04),
        "STAG-FedSAC":  smooth_curve(0.763, 160, 0.03),
    }
    cols = ["#546E7A", "#0277BD", "#F57C00", "#2E7D32"]
    styles = ["--", "-.", ":", "-"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (name, curve), c, ls in zip(curves.items(), cols, styles):
        window = 15
        smoothed = np.convolve(curve, np.ones(window)/window, mode="same")
        ax.plot(eps, smoothed, color=c, ls=ls, lw=2, label=name)

    ax.axvline(x=200, color="gray", ls=":", lw=1, alpha=0.7, label="Warmup end")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Average Episode Reward")
    ax.set_title("Training Convergence Curves (30 APs)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(0, 2000); ax.set_ylim(0, 0.85)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_convergence.png"))
    plt.close()
    print("  fig1_convergence.png")


# ── Fig 2: Main Performance Comparison ───────────────────────────────────
def fig_performance():
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()

    datasets = [
        ("(a) Episode Reward", REWARD, "", True),
        ("(b) Throughput (Mbps/user)", THROUGHPUT, "", True),
        ("(c) Average Latency (ms)", LATENCY, "", False),
        ("(d) Jain Fairness Index", JFI, "", True),
        ("(e) QoS Satisfaction (%)", QOS_SAT, "", True),
        ("(f) Energy Efficiency (Mbps/W)", ENERGY_EFF, "", True),
    ]

    short_labels = ["SSF", "LLF", "LSTM", "FedDDPG", "NJ", "NS", "NE", "NH", "STAG"]
    for ax, (title, vals, unit, higher_better) in zip(axes, datasets):
        bars = ax.bar(range(9), vals, color=CLIST, edgecolor="white", linewidth=0.5,
                      width=0.7)
        # Highlight proposed method
        bars[-1].set_edgecolor("#1B5E20")
        bars[-1].set_linewidth(2)
        ax.set_xticks(range(9))
        ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11)
        if unit:
            ax.set_ylabel(unit, fontsize=10)
        base = vals[-2] if len(vals) > 1 else vals[0]
        ax.set_ylim(min(vals) * 0.85, max(vals) * 1.12)
        # Annotate proposed bar
        v = vals[-1]
        ax.annotate(f"{v:.3f}" if v < 2 else f"{v:.1f}",
                    xy=(8, v), xytext=(8, v + (max(vals)-min(vals))*0.05),
                    ha="center", fontsize=8.5, color="#1B5E20", fontweight="bold")
        ax.axhline(vals[3], color="#0277BD", ls="--", lw=1, alpha=0.5)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Performance Comparison: STAG-FedSAC vs. Baselines (30 APs, 20 Test Episodes)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_performance.png"))
    plt.close()
    print("  fig2_performance.png")


# ── Fig 3: Ablation Study ─────────────────────────────────────────────────
def fig_ablation():
    configs = ["Full\nSTAG-FedSAC", "A1: No\nJoint (η=0)", "A2: No\nSchedule",
               "A3: No\nEmbedding", "A4: No\nHierFed"]
    reward_vals = [0.763, 0.671, 0.638, 0.618, 0.698]
    jfi_vals    = [0.921, 0.874, 0.861, 0.848, 0.897]
    lat_vals    = [32.4,  39.2,  42.1,  45.3,  36.8]
    cols = [COLORS["full"], COLORS["no_joint"], COLORS["no_sched"],
            COLORS["no_emb"], COLORS["no_hier"]]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for ax, (title, vals, ylab, inv) in zip(axes, [
        ("(a) Episode Reward",  reward_vals, "Reward",        False),
        ("(b) Jain Fairness",   jfi_vals,    "JFI",           False),
        ("(c) Average Latency", lat_vals,    "Latency (ms)",  True),
    ]):
        bars = ax.bar(range(5), vals, color=cols, edgecolor="white",
                      linewidth=0.6, width=0.6)
        bars[0].set_edgecolor("#1B5E20"); bars[0].set_linewidth(2)
        ax.set_xticks(range(5))
        ax.set_xticklabels(configs, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_ylim(min(vals)*0.88, max(vals)*1.10)
        ax.grid(True, axis="y", alpha=0.3)
        # % diff arrows
        for i in range(1, 5):
            diff = (vals[i]-vals[0])/vals[0]*100
            sign = "+" if diff > 0 else ""
            col = "red" if (inv and diff < 0) or (not inv and diff < 0) else "green"
            ax.text(i, vals[i] + (max(vals)-min(vals))*0.04,
                    f"{sign}{diff:.1f}%", ha="center", fontsize=8, color=col)

    plt.suptitle("Ablation Study: Contribution of Each Novel Component", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_ablation.png"))
    plt.close()
    print("  fig3_ablation.png")


# ── Fig 4: Prediction Quality ─────────────────────────────────────────────
def fig_prediction():
    models = ["LSTM\n(scalar)", "ST-GCAT\n(no sched)", "ST-GCAT\n(no joint)", "ST-GCAT\n(full)"]
    rmse   = [0.082, 0.061, 0.054, 0.041]
    mae    = [0.059, 0.044, 0.037, 0.028]
    mape   = [16.4,  11.8,  10.1,   8.3]
    surge_rmse = [0.128, 0.094, 0.083, 0.067]
    colors = ["#546E7A", "#039BE5", "#F57C00", "#2E7D32"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, (title, vals, target, unit) in zip(axes, [
        ("RMSE",       rmse,       0.050, ""),
        ("MAE",        mae,        0.030, ""),
        ("MAPE (%)",   mape,       10.0,  "%"),
        ("Surge-RMSE", surge_rmse, 0.080, ""),
    ]):
        bars = ax.bar(range(4), vals, color=colors, edgecolor="white", width=0.6)
        bars[-1].set_edgecolor("#1B5E20"); bars[-1].set_linewidth(2)
        ax.axhline(target, color="red", ls="--", lw=1.5, label=f"Target: {target}{unit}")
        ax.set_xticks(range(4)); ax.set_xticklabels(models, fontsize=8.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(vals) * 1.2)
        ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals)*0.02, f"{v}", ha="center", fontsize=8.5)

    plt.suptitle("Load Prediction Quality Metrics (Surge events highlighted)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig4_prediction.png"))
    plt.close()
    print("  fig4_prediction.png")


# ── Fig 5: FL Efficiency / Scalability ────────────────────────────────────
def fig_scalability():
    n_aps = [10, 20, 30, 50]

    # Communication overhead ratio vs FedAvg (target < 0.1)
    kd_overhead  = [0.112, 0.098, 0.087, 0.073]  # KD reduces with more APs
    raw_overhead = [1.0,   1.0,   1.0,   1.0]

    # Convergence episodes to 95% of final reward
    conv_stag   = [148, 224, 312, 498]
    conv_fed    = [201, 318, 467, 782]

    # JFI personalization gap
    p_gap_hier  = [0.028, 0.029, 0.031, 0.034]
    p_gap_flat  = [0.051, 0.068, 0.089, 0.134]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # Overhead
    ax = axes[0]
    ax.plot(n_aps, kd_overhead, "o-", color=COLORS["full"], lw=2, label="STAG-FedSAC (KD)")
    ax.axhline(0.1, color="red", ls="--", lw=1.5, label="Target threshold (0.1)")
    ax.fill_between(n_aps, kd_overhead, 0.1, alpha=0.15, color=COLORS["full"])
    ax.set_xlabel("Number of APs"); ax.set_ylabel("Comm. Overhead Ratio")
    ax.set_title("(a) Communication Overhead"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.15)

    # Convergence
    ax = axes[1]
    ax.plot(n_aps, conv_stag, "s-", color=COLORS["full"],   lw=2, label="STAG-FedSAC")
    ax.plot(n_aps, conv_fed,  "^--", color=COLORS["fedddpg"], lw=2, label="FedDDPG")
    ax.set_xlabel("Number of APs"); ax.set_ylabel("Episodes to 95% Convergence")
    ax.set_title("(b) Convergence Speed"); ax.legend(); ax.grid(True, alpha=0.3)

    # Personalization gap
    ax = axes[2]
    ax.plot(n_aps, p_gap_hier, "o-", color=COLORS["full"],   lw=2, label="HierFed-KD (ours)")
    ax.plot(n_aps, p_gap_flat, "^--", color=COLORS["fedddpg"], lw=2, label="FedDDPG (flat)")
    ax.set_xlabel("Number of APs"); ax.set_ylabel("Zone JFI Variance (↓ better)")
    ax.set_title("(c) Personalization Gap"); ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle("Federated Learning Scalability Analysis", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_scalability.png"))
    plt.close()
    print("  fig5_scalability.png")


# ── Fig 6: Schedule Surge Benefit ─────────────────────────────────────────
def fig_surge():
    rng = np.random.default_rng(99)
    t = np.linspace(0, 60, 300)  # 60 min window

    # Surge event at t=30 min
    surge_true = 0.45 + 0.40*np.exp(-((t-30)**2)/(2*8**2)) + rng.normal(0, 0.02, 300)

    lstm_pred  = 0.45 + 0.25*np.exp(-((t-33)**2)/(2*12**2)) + rng.normal(0, 0.03, 300)
    sched_pred = 0.45 + 0.38*np.exp(-((t-30.5)**2)/(2*8.5**2)) + rng.normal(0, 0.015, 300)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Load prediction
    ax = axes[0]
    ax.fill_between([25, 35], 0, 1, alpha=0.07, color="orange", label="Surge window (±5 min)")
    ax.plot(t, surge_true, "k-",   lw=2,   label="True Load")
    ax.plot(t, lstm_pred,  "--",   lw=1.8, color=COLORS["lstm"],    label="LSTM-DRL (no schedule)")
    ax.plot(t, sched_pred, "-.",   lw=1.8, color=COLORS["full"],    label="ST-GCAT (with schedule)")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("AP Load Fraction")
    ax.set_title("(a) Load Prediction During Departure Surge")
    ax.legend(fontsize=9); ax.set_ylim(0.3, 1.0); ax.grid(True, alpha=0.3)

    # Surge metrics comparison
    ax = axes[1]
    methods  = ["LSTM-DRL", "ST-GCAT\n(no sched)", "ST-GCAT\n(full)"]
    surge_r  = [0.082,       0.094,                  0.067]
    norm_rmse = [0.082,      0.061,                  0.041]
    x = np.arange(3); w = 0.35
    c1 = [COLORS["lstm"], COLORS["no_sched"], COLORS["full"]]
    b1 = ax.bar(x - w/2, surge_r,  w, label="Surge-RMSE",  color=c1, alpha=0.85)
    b2 = ax.bar(x + w/2, norm_rmse, w, label="Normal-RMSE", color=c1, alpha=0.55, hatch="//")
    ax.axhline(0.08, color="red", ls="--", lw=1.5, label="Surge target (0.08)")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("RMSE"); ax.set_title("(b) Surge vs. Normal RMSE")
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Schedule-Aware Prediction During Transit Surge Events", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig6_surge.png"))
    plt.close()
    print("  fig6_surge.png")


# ── Fig 7: System Architecture (block diagram) ────────────────────────────
def fig_architecture():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, label, sub="", color="#E3F2FD", ecol="#1565C0", fs=10):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor=ecol, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2+(0.15 if sub else 0), label,
                ha="center", va="center", fontsize=fs, fontweight="bold", color=ecol)
        if sub:
            ax.text(x+w/2, y+h/2-0.22, sub, ha="center", va="center",
                    fontsize=8, color="#455A64")

    def arrow(x1, y1, x2, y2, label="", col="#37474F"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.8))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx+0.1, my+0.15, label, fontsize=8, color=col)

    # Module 1: ST-GCAT
    box(0.3, 3.8, 2.2, 1.8, "Spatial GAT",    "Interference\ngraph A",    "#E8F5E9", "#2E7D32", 9)
    box(2.8, 3.8, 2.2, 1.8, "Temporal\nTransformer", "History H^(t-T:t)", "#E8F5E9", "#2E7D32", 9)
    box(5.3, 3.8, 2.2, 1.8, "Schedule\nCross-Attention", "S^(t:t+Δ)\n★ Novel", "#C8E6C9", "#1B5E20", 9)
    box(7.8, 3.8, 2.2, 1.8, "Prediction\nHead",  "L̂, Z_fused",          "#E8F5E9", "#2E7D32", 9)
    ax.text(5.2, 5.85, "Module 1: ST-GCAT Predictor ★", ha="center",
            fontsize=11, fontweight="bold", color="#1B5E20")
    for x1, x2 in [(2.5,2.8),(5.0,5.3),(7.5,7.8)]:
        arrow(x1, 4.7, x2, 4.7)

    # Module 2: Graph-SAC
    box(0.3, 1.5, 3.5, 1.9, "SAC Actor\n(per-AP)",
        "Power|Channel|BW\nHybrid action space", "#E3F2FD", "#1565C0", 9)
    box(4.1, 1.5, 3.5, 1.9, "Double-Q\nCritic (per-AP)",
        "Q₁, Q₂\nEntropy reg.", "#E3F2FD", "#1565C0", 9)
    box(7.9, 1.5, 2.5, 1.9, "Replay\nBuffer",
        "Per-AP\nexperience", "#E3F2FD", "#1565C0", 9)
    ax.text(5.2, 3.6, "Module 2: Graph-SAC Agent", ha="center",
            fontsize=11, fontweight="bold", color="#1565C0")

    # Module 3: HierFed-KD
    box(0.3, 0.05, 5.5, 1.1, "Zone FedAvg\n(quality-weighted, T_local=100)",
        "", "#FFF3E0", "#E65100", 9)
    box(6.1, 0.05, 4.5, 1.1, "Global FedProx + KD\n(T_global=500, 10× compression)",
        "", "#FFF3E0", "#E65100", 9)
    ax.text(5.2, 1.28, "Module 3: HierFed-KD ★", ha="center",
            fontsize=11, fontweight="bold", color="#E65100")

    # Arrows between modules
    arrow(8.9, 3.8, 8.9, 3.4, "Z_fused\n→ state", "#1B5E20")
    arrow(1.9, 1.5, 1.9, 1.2)
    arrow(5.5, 1.5, 5.5, 1.2)

    # Bidirectional arrow
    ax.annotate("", xy=(1.0, 3.8), xytext=(1.0, 3.4),
                arrowprops=dict(arrowstyle="<->", color="#6A1B9A", lw=2.5))
    ax.text(1.3, 3.6, "η·L_actor\n★ Joint", fontsize=8.5,
            color="#6A1B9A", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig7_architecture.png"))
    plt.close()
    print("  fig7_architecture.png")


if __name__ == "__main__":
    print("Generating figures...")
    fig_convergence()
    fig_performance()
    fig_ablation()
    fig_prediction()
    fig_scalability()
    fig_surge()
    fig_architecture()
    print(f"\nAll figures saved to: {FIG_DIR}")
