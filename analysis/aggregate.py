"""
Aggregate raw experiment records (results/raw/*.json) into the LaTeX tables, figures and
statistics used in the manuscript. Nothing here is typed in by hand: every number in the
paper's tables and figures is computed from the logged runs.

    python analysis/aggregate.py --raw results/raw --out-tables manuscripts/tables \
                                 --out-figs manuscripts/figures --summary results/summary.json
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {
    "ssf": "SSF", "llf": "LLF", "persist": "Persistence", "lstm_drl": "LSTM-DRL", "fedddpg": "FedDDPG",
    "full": "STAG-FedSAC", "a1_nojoint": "A1: no joint gradient", "a2_nosched": "A2: no schedule",
    "a3_noemb": "A3: no embedding", "a4_zone": "Zone FedAvg only", "a5_hieravg": "Hier. FedAvg+FedProx (weights)",
    "iso": "Isolated (no federation)", "flat": "Flat FedAvg", "central": "Pooled experience (centralised)",
}
MAIN_ORDER = ["ssf", "llf", "lstm_drl", "fedddpg", "a1_nojoint", "a2_nosched", "a3_noemb", "full"]
FED_ORDER = ["iso", "flat", "a4_zone", "a5_hieravg", "full", "central"]
METRICS = [("reward", "Reward", 3), ("tput", "Thrpt. (Mbps/user)", 2), ("lat", "Lat. (ms)", 1),
           ("jfi", "JFI", 3), ("qos", "QoS sat.", 3), ("eeff", "Mbps/W", 2), ("switch", "Sw./AP/step", 3)]


def load(raw_dir: str):
    runs = defaultdict(lambda: defaultdict(dict))        # tag -> method -> seed -> record
    for f in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
        m = re.match(r"(?P<tag>[a-z0-9]+)_(?P<method>[a-z0-9_]+)_s(?P<seed>\d+)\.json", os.path.basename(f))
        if not m:
            continue
        with open(f) as fh:
            runs[m["tag"]][m["method"]][int(m["seed"])] = json.load(fh)
    return runs


def seed_values(recs: dict, metric: str, cond: str = "main") -> dict[int, float]:
    return {s: r["tests"][cond]["summary"][metric] for s, r in recs.items()
            if metric in r["tests"][cond]["summary"]}


def fmt(vals, d: int) -> str:
    v = np.array(list(vals), dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return "--"
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{v.mean():.{d}f}" + (f"{{\\tiny$\\pm${sd:.{d}f}}}" if len(v) > 1 else "")


def signflip_p(diffs: np.ndarray) -> float:
    """Exact two-sided paired sign-flip permutation test on the mean difference."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n < 2:
        return float("nan")
    obs = abs(d.mean())
    cnt = 0
    tot = 0
    for signs in itertools.product([-1, 1], repeat=n):
        tot += 1
        if abs((d * np.array(signs)).mean()) >= obs - 1e-12:
            cnt += 1
    return cnt / tot


def boot_ci(d: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    d = np.asarray(d, dtype=float)
    means = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(recs_a: dict, recs_b: dict, metric: str, cond: str = "main"):
    a, b = seed_values(recs_a, metric, cond), seed_values(recs_b, metric, cond)
    common = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in common])
    return d, common


def write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def table_rows(methods, recs_by_method, metrics, best_bold=True):
    cols = {}
    for key, _, d in metrics:
        cols[key] = {m: np.nanmean(list(seed_values(recs_by_method[m], key).values())) for m in methods if m in recs_by_method and recs_by_method[m]}
    lines = []
    for m in methods:
        if m not in recs_by_method or not recs_by_method[m]:
            continue
        cells = []
        for key, _, d in metrics:
            vals = seed_values(recs_by_method[m], key).values()
            cell = fmt(vals, d)
            if best_bold and key in ("reward", "tput", "jfi", "qos", "eeff") and m not in ("persist",):
                cand = {k: v for k, v in cols[key].items() if k != "persist"}
                if cand and np.isclose(cols[key][m], max(cand.values())):
                    cell = f"\\textbf{{{cell}}}"
            if best_bold and key in ("lat", "switch") and m not in ("persist",):
                cand = {k: v for k, v in cols[key].items() if k != "persist"}
                if cand and np.isclose(cols[key][m], min(cand.values())):
                    cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        n_seeds = len(recs_by_method[m])
        lines.append(f"{NAMES[m]} & {n_seeds} & " + " & ".join(cells) + " \\\\")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--out-tables", default="manuscripts/tables")
    ap.add_argument("--out-figs", default="manuscripts/figures")
    ap.add_argument("--summary", default="results/summary.json")
    args = ap.parse_args()
    runs = load(args.raw)
    main_runs = runs.get("main", {})
    summary: dict = {"n_runs": {t: {m: len(s) for m, s in d.items()} for t, d in runs.items()}}

    # ---- Table: main comparison --------------------------------------------------
    head = "Method & Seeds & " + " & ".join(n for _, n, _ in METRICS) + " \\\\"
    lines = table_rows(MAIN_ORDER, main_runs, METRICS)
    write(os.path.join(args.out_tables, "main_results.tex"),
          "\\begin{tabular}{llccccccc}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- Statistics: full vs each other method ---------------------------------------
    stats_rows = []
    summary["tests"] = {}
    if "full" in main_runs:
        others = [m for m in MAIN_ORDER if m != "full" and m in main_runs]
        n_cmp = max(1, len(others))
        for m in others:
            row = {"method": m}
            for key in ("reward", "lat", "jfi", "qos"):
                d, common = paired(main_runs["full"], main_runs[m], key)
                if len(d) >= 2:
                    p = signflip_p(d)
                    lo, hi = boot_ci(d)
                    row[key] = dict(mean_diff=float(d.mean()), ci=(lo, hi), p=p, p_bonf=min(1.0, p * n_cmp), n=len(d))
            summary["tests"][m] = row
            if "reward" in row:
                r = row["reward"]
                stats_rows.append(f"{NAMES[m]} & {r['mean_diff']:+.3f} & [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] & "
                                  f"{r['p']:.3f} & {r['p_bonf']:.3f} & {r['n']} \\\\")
        write(os.path.join(args.out_tables, "stats.tex"),
              "\\begin{tabular}{lcccrc}\n\\toprule\nSTAG-FedSAC vs. & $\\Delta$ reward & 95\\% CI & $p$ & $p_{\\text{Bonf}}$ & Seeds \\\\\n\\midrule\n"
              + "\n".join(stats_rows) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- Table: prediction quality -----------------------------------------------------
    pm = [("rmse", "RMSE", 3), ("mae", "MAE", 3), ("rmse_h1", "RMSE ($h{=}1$)", 3), ("surge_rmse", "Surge-RMSE", 3),
          ("normal_rmse", "Normal-RMSE", 3), ("mape", "MAPE (\\%)", 1)]
    pred_methods = ["persist", "lstm_drl", "a2_nosched", "a1_nojoint", "full"]
    pl = []
    for m in pred_methods:
        if m in main_runs and main_runs[m]:
            pl.append(f"{NAMES[m]} & " + " & ".join(fmt(seed_values(main_runs[m], k).values(), d) for k, _, d in pm) + " \\\\")
    write(os.path.join(args.out_tables, "prediction.tex"),
          "\\begin{tabular}{l" + "c" * len(pm) + "}\n\\toprule\nForecaster & " + " & ".join(n for _, n, _ in pm) +
          " \\\\\n\\midrule\n" + "\n".join(pl) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- Table: federation study + communication -----------------------------------------
    fed_runs = {m: dict(main_runs.get(m, {})) for m in FED_ORDER}
    fm = [("reward", "Reward", 3), ("lat", "Lat. (ms)", 1), ("jfi", "JFI", 3), ("qos", "QoS sat.", 3)]
    fl = []
    comm_summary = {}
    for m in FED_ORDER:
        if not fed_runs.get(m):
            continue
        comm = [r["history"].get("comm", {}) for r in fed_runs[m].values() if r["history"].get("comm")]
        wan = np.mean([c["wan_total"] for c in comm]) / 1e6 if comm else 0.0
        intra = np.mean([c["intra_zone"] for c in comm]) / 1e6 if comm else 0.0
        comm_summary[m] = dict(wan_MB=float(wan), intra_MB=float(intra))
        fl.append(f"{NAMES[m]} & {len(fed_runs[m])} & " + " & ".join(fmt(seed_values(fed_runs[m], k).values(), d) for k, _, d in fm)
                  + f" & {intra:.1f} & {wan:.1f} \\\\")
    write(os.path.join(args.out_tables, "federation.tex"),
          "\\begin{tabular}{llcccccc}\n\\toprule\nScheme & Seeds & Reward & Lat. (ms) & JFI & QoS sat. & Intra-zone (MB) & Inter-zone (MB) \\\\\n\\midrule\n"
          + "\n".join(fl) + "\n\\bottomrule\n\\end{tabular}\n")
    summary["comm"] = comm_summary

    # ---- Table: robustness (MAC contention / switching cost) ---------------------------------
    rob = []
    conds = [("main", "Full MAC, 3 s outage"), ("no_mac", "No contention"), ("no_switch_cost", "No switch cost"), ("switch_10s", "10 s outage")]
    for m in ["ssf", "llf", "fedddpg", "full"]:
        if m in main_runs and main_runs[m]:
            cells = []
            for c, _ in conds:
                cells.append(fmt(seed_values(main_runs[m], "reward", c).values(), 3) + " / " + fmt(seed_values(main_runs[m], "lat", c).values(), 0))
            rob.append(f"{NAMES[m]} (trained w/ MAC) & " + " & ".join(cells) + " \\\\")
    nomac = runs.get("nomac", {})
    for m in ["fedddpg", "full"]:
        if m in nomac and nomac[m]:
            cells = [fmt(seed_values(nomac[m], "reward", c).values(), 3) + " / " + fmt(seed_values(nomac[m], "lat", c).values(), 0) for c, _ in conds]
            rob.append(f"{NAMES[m]} (trained w/o MAC) & " + " & ".join(cells) + " \\\\")
    out0 = runs.get("out0", {})
    for m in ["full"]:
        if m in out0 and out0[m]:
            cells = [fmt(seed_values(out0[m], "reward", c).values(), 3) + " / " + fmt(seed_values(out0[m], "lat", c).values(), 0) for c, _ in conds]
            rob.append(f"{NAMES[m]} (trained w/o switch cost) & " + " & ".join(cells) + " \\\\")
    write(os.path.join(args.out_tables, "robustness.tex"),
          "\\begin{tabular}{lcccc}\n\\toprule\nPolicy & " + " & ".join(n for _, n in conds) + " \\\\\n\\midrule\n" + "\n".join(rob) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- Table: reward-weight sensitivity ----------------------------------------------------
    wl = []
    wnames = {"w1": "Equal weights (0.25/0.25/0.25/0.25)", "w2": "Throughput-heavy (0.55/0.15/0.20/0.10)", "w3": "Latency-heavy (0.20/0.50/0.20/0.10)"}
    for tag in ("w1", "w2", "w3"):
        r = runs.get(tag, {})
        if r.get("full") and r.get("fedddpg"):
            d, _ = paired(r["full"], r["fedddpg"], "reward")
            wl.append(f"{wnames[tag]} & " + " & ".join(
                f"{fmt(seed_values(r[m], k).values(), dd)}" for m in ("fedddpg", "full") for k, dd in (("reward", 3), ("lat", 0), ("jfi", 3))) + " \\\\")
    if wl:
        write(os.path.join(args.out_tables, "weights.tex"),
              "\\begin{tabular}{lcccccc}\n\\toprule\n & \\multicolumn{3}{c}{FedDDPG} & \\multicolumn{3}{c}{STAG-FedSAC} \\\\\n"
              "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nWeights ($\\alpha/\\beta/\\gamma/\\delta$) & Reward & Lat. & JFI & Reward & Lat. & JFI \\\\\n\\midrule\n"
              + "\n".join(wl) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- Table: weight of the joint gradient ------------------------------------------------------
    etarows = []
    eta_sets = [("0", main_runs.get("a1_nojoint", {})), ("0.01", runs.get("eta001", {}).get("full", {})), ("0.1", main_runs.get("full", {}))]
    for lab, rr in eta_sets:
        if rr:
            etarows.append(f"{lab} & {len(rr)} & " + " & ".join(fmt(seed_values(rr, k).values(), d) for k, d in (("reward", 3), ("lat", 1), ("rmse", 3), ("surge_rmse", 3))) + " \\\\")
    if etarows:
        write(os.path.join(args.out_tables, "eta.tex"),
              "\\begin{tabular}{llcccc}\n\\toprule\n$\\eta$ & Seeds & Reward & Lat. (ms) & RMSE & Surge-RMSE \\\\\n\\midrule\n" + "\n".join(etarows) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- Figures -------------------------------------------------------------------------------
    os.makedirs(args.out_figs, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "savefig.dpi": 300, "savefig.bbox": "tight"})
    col = {"ssf": "#9E9E9E", "llf": "#78909C", "lstm_drl": "#546E7A", "fedddpg": "#0277BD", "a1_nojoint": "#F57C00",
           "a2_nosched": "#E53935", "a3_noemb": "#8E24AA", "full": "#2E7D32"}
    # convergence (evaluation reward on 5 held-out episodes, mean +- std over seeds)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for m in ["lstm_drl", "fedddpg", "a1_nojoint", "full"]:
        rs = main_runs.get(m, {})
        curves = [(r["history"]["eval_episode"], r["history"]["eval_reward"]) for r in rs.values() if "eval_reward" in r["history"]]
        if not curves:
            continue
        n = min(len(c[0]) for c in curves)
        x = np.array(curves[0][0][:n])
        y = np.array([c[1][:n] for c in curves])
        ax.plot(x, y.mean(0), color=col[m], label=NAMES[m], lw=1.8)
        ax.fill_between(x, y.mean(0) - y.std(0), y.mean(0) + y.std(0), color=col[m], alpha=0.15)
    for m, ls in (("ssf", ":"), ("llf", "--")):
        if main_runs.get(m):
            ax.axhline(np.mean(list(seed_values(main_runs[m], "reward").values())), color=col[m], ls=ls, lw=1.2, label=NAMES[m] + " (no learning)")
    ax.axvline(200, color="k", lw=0.6, ls=":")
    ax.text(205, ax.get_ylim()[0], "end of random warm-up", fontsize=8, va="bottom")
    ax.set_xlabel("Training episode"); ax.set_ylabel("Held-out reward (5 episodes)")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(args.out_figs, "fig_convergence.png")); plt.close(fig)

    # main metrics bars
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
    for ax, (k, n, d) in zip(axes, [m for m in METRICS if m[0] in ("reward", "lat", "jfi", "qos")]):
        ms = [m for m in MAIN_ORDER if main_runs.get(m)]
        mu = [np.mean(list(seed_values(main_runs[m], k).values())) for m in ms]
        sd = [np.std(list(seed_values(main_runs[m], k).values()), ddof=1) if len(main_runs[m]) > 1 else 0 for m in ms]
        ax.bar(range(len(ms)), mu, yerr=sd, color=[col[m] for m in ms], capsize=2)
        ax.set_xticks(range(len(ms))); ax.set_xticklabels([NAMES[m].split(":")[0] for m in ms], rotation=60, ha="right", fontsize=8)
        ax.set_title(n); ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(args.out_figs, "fig_main.png")); plt.close(fig)

    # federation: reward vs WAN traffic
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for m in FED_ORDER:
        if fed_runs.get(m):
            r = np.array(list(seed_values(fed_runs[m], "reward").values()))
            w = comm_summary.get(m, {}).get("wan_MB", 0.0)
            ax.errorbar(max(w, 0.05), r.mean(), yerr=r.std(ddof=1) if len(r) > 1 else 0, fmt="o", capsize=3, label=NAMES[m])
    ax.set_xscale("log"); ax.set_xlabel("Inter-zone traffic during training (MB, log scale; 0 plotted at 0.05)")
    ax.set_ylabel("Held-out reward"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.savefig(os.path.join(args.out_figs, "fig_federation.png")); plt.close(fig)

    # prediction by surge / normal
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ms = [m for m in pred_methods if main_runs.get(m)]
    xs = np.arange(len(ms))
    for off, (k, lab, hatch) in zip((-0.2, 0.2), (("normal_rmse", "Normal", "//"), ("surge_rmse", "Surge", ""))):
        vals = [np.nanmean(list(seed_values(main_runs[m], k).values())) for m in ms]
        ax.bar(xs + off, vals, 0.4, label=lab, hatch=hatch, color="#4C72B0" if k == "normal_rmse" else "#C44E52")
    ax.set_xticks(xs); ax.set_xticklabels([NAMES[m].replace("Persistence", "Persist.") for m in ms], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("RMSE (all horizons)"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(args.out_figs, "fig_prediction.png")); plt.close(fig)

    # ---- LaTeX macros for numbers quoted in the text -----------------------------------------
    MN = {"ssf": "SSF", "llf": "LLF", "persist": "Persist", "lstm_drl": "LSTMDRL", "fedddpg": "FedDDPG", "full": "Full",
          "a1_nojoint": "AOne", "a2_nosched": "ATwo", "a3_noemb": "AThree", "a4_zone": "AFour", "a5_hieravg": "AFive",
          "iso": "Iso", "flat": "Flat", "central": "Central"}
    KN = {"reward": "Reward", "tput": "Tput", "lat": "Lat", "jfi": "Jfi", "qos": "Qos", "eeff": "Eeff", "switch": "Switch",
          "rmse": "Rmse", "mae": "Mae", "rmse_h1": "RmseHOne", "surge_rmse": "SurgeRmse", "normal_rmse": "NormalRmse", "mape": "Mape",
          "util": "Util"}
    mac = []
    for tag, tname in (("main", ""), ("nomac", "NoMac"), ("out0", "OutZero"), ("w1", "WOne"), ("w2", "WTwo"), ("w3", "WThree"), ("eta001", "EtaSmall")):
        for m, recs in runs.get(tag, {}).items():
            if not recs:
                continue
            mac.append("\\newcommand{\\n%s%s}{%d}" % (tname, MN[m], len(recs)))
            for cond, cname in (("main", ""), ("no_mac", "NoMacEval"), ("no_switch_cost", "NoSwEval"), ("switch_10s", "SwTenEval")):
                for k, kn in KN.items():
                    v = np.array(list(seed_values(recs, k, cond).values()), dtype=float)
                    v = v[~np.isnan(v)]
                    if len(v):
                        d = 1 if k in ("lat", "mape") else (2 if k in ("tput", "eeff") else 3)
                        mac.append("\\newcommand{\\m%s%s%s%s}{%.*f}" % (tname, cname, MN[m], kn, d, v.mean()))
                        mac.append("\\newcommand{\\s%s%s%s%s}{%.*f}" % (tname, cname, MN[m], kn, d, (v.std(ddof=1) if len(v) > 1 else 0)))
    for m, row in summary.get("tests", {}).items():
        for k, r in row.items():
            if k == "method":
                continue
            kn = KN[k]
            mac.append("\\newcommand{\\dFull%s%s}{%+.3f}" % (MN[m], kn, r["mean_diff"]))
            mac.append("\\newcommand{\\ciLoFull%s%s}{%+.3f}" % (MN[m], kn, r["ci"][0]))
            mac.append("\\newcommand{\\ciHiFull%s%s}{%+.3f}" % (MN[m], kn, r["ci"][1]))
            mac.append("\\newcommand{\\pFull%s%s}{%.3f}" % (MN[m], kn, r["p"]))
    for m, c in comm_summary.items():
        mac.append("\\newcommand{\\commWan%s}{%.2f}" % (MN[m], c["wan_MB"]))
        mac.append("\\newcommand{\\commIntra%s}{%.1f}" % (MN[m], c["intra_MB"]))
    if "full" in comm_summary and comm_summary["full"]["wan_MB"] > 0:
        for m in ("flat", "a5_hieravg"):
            if m in comm_summary and comm_summary[m]["wan_MB"] > 0:
                mac.append("\\newcommand{\\commRatio%s}{%.0f}" % (MN[m], comm_summary[m]["wan_MB"] / comm_summary["full"]["wan_MB"]))
        rr = [r["history"]["comm"] for r in main_runs.get("full", {}).values() if r["history"].get("comm")]
        if rr:
            mac.append("\\newcommand{\\commRefMB}{%.1f}" % (np.mean([c["reference_set"] for c in rr]) / 1e6))
            mac.append("\\newcommand{\\commOutMB}{%.2f}" % (np.mean([c["inter_zone"] for c in rr]) / 1e6))
    # derived quantities: relative reductions and the oracle reference
    mm = summary.get("main_means", {})
    def mean_of(tag, m, k, cond="main"):
        r = runs.get(tag, {}).get(m, {})
        v = [x for x in seed_values(r, k, cond).values() if not np.isnan(x)]
        return float(np.mean(v)) if v else None
    for m in ("persist", "lstm_drl", "a2_nosched", "a1_nojoint", "a3_noemb"):
        for k, kn in (("rmse", "Rmse"), ("surge_rmse", "SurgeRmse"), ("normal_rmse", "NormalRmse")):
            a, b = mean_of("main", "full", k), mean_of("main", m, k)
            if a is not None and b is not None and b > 0:
                mac.append("\\newcommand{\\redFull%s%s}{%.0f}" % (MN[m], kn, 100 * (b - a) / b))
    oracle_path = os.path.join(os.path.dirname(args.raw.rstrip("/")), "oracle.json")
    if os.path.exists(oracle_path):
        o = json.load(open(oracle_path))
        mac.append("\\newcommand{\\oracleReward}{%.3f}" % o["oracle_mean"])
        mac.append("\\newcommand{\\oracleSSFReward}{%.3f}" % o["ssf_mean"])
        mac.append("\\newcommand{\\oracleGainPct}{%.0f}" % (100 * (o["oracle_mean"] - o["ssf_mean"]) / o["ssf_mean"]))
        mac.append("\\newcommand{\\oracleEpisodes}{%d}" % o["episodes"])
    write(os.path.join(args.out_tables, "numbers.tex"), "% generated by analysis/aggregate.py\n" + "\n".join(mac) + "\n")

    summary["main"] = {m: {k: float(np.nanmean(list(seed_values(r, k).values()))) for k, _, _ in METRICS + pm if seed_values(r, k)}
                       for m, r in main_runs.items()}
    write(args.summary, json.dumps(summary, indent=2))
    print(json.dumps(summary["n_runs"], indent=1))


if __name__ == "__main__":
    main()
