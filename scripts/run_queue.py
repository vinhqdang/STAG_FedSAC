"""
Run a list of experiment jobs, two at a time, skipping jobs whose output already exists.

    python scripts/run_queue.py jobs.json results/raw [--concurrency 2]

jobs.json: list of {"tag": str, "method": str, "seed": int, "args": [extra CLI args]}.
Output files are named <tag>_<method>_s<seed>.json (read by analysis/aggregate.py).
"""
import argparse
import json
import os
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("jobs")
ap.add_argument("out")
ap.add_argument("--concurrency", type=int, default=2)
ap.add_argument("--episodes", type=int, default=2000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
jobs = json.load(open(a.jobs))
running: list[tuple[subprocess.Popen, str]] = []


def launch(j):
    name = f"{j['tag']}_{j['method']}_s{j['seed']}"
    out = os.path.join(a.out, name + ".json")
    if os.path.exists(out):
        return None
    cmd = [sys.executable, "-m", "stag_fedsac.evaluation.run_experiment", "--method", j["method"], "--seed", str(j["seed"]),
           "--episodes", str(a.episodes), "--out", out] + [str(x) for x in j.get("args", [])]
    log = open(os.path.join(a.out, name + ".log"), "w")
    print(time.strftime("%H:%M:%S"), "start", name, flush=True)
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), name


pending = list(jobs)
while pending or running:
    running = [(p, n) for p, n in running if p.poll() is None or print(time.strftime("%H:%M:%S"), "done", n, flush=True)]
    while pending and len(running) < a.concurrency:
        r = launch(pending.pop(0))
        if r:
            running.append(r)
    time.sleep(5)
print("ALL DONE", flush=True)
