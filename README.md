# STAG-FedSAC

Schedule-conditioned spatio-temporal graph attention (ST-GCAT) forecasting, per-AP Soft Actor-Critic agents and
hierarchical federation (HierFed-KD) for WiFi resource management in transit venues, evaluated in a simulator
with MAC contention (Bianchi model), a channel-switching cost and an EDCA-style airtime mapping.

**Scope.** All results are obtained in simulation. No measured traces are used (the dataset loaders in
`stag_fedsac/data/` are not used by any reported experiment). Every number, table and figure of the manuscript
is computed from the run records in `results/raw/` by `analysis/aggregate.py`.

## Layout

```
stag_fedsac/
  environment/   wifi_env.py (simulator), mac_model.py (Bianchi efficiency, airtime quantiser), schedule_generator.py
  models/        stgcat.py (predictor), graph_sac.py (batched per-AP actors/critics), hierfed_kd.py (federation, byte counter)
  training/      joint_trainer.py (STAG-FedSAC and ablations), common.py (episodes, evaluation), replay_buffer.py
  evaluation/    baselines.py (SSF, LLF, LSTM-DRL, FedDDPG, persistence), run_experiment.py (one method/seed -> JSON)
scripts/         run_queue.py (job queue), sync.sh
analysis/        aggregate.py (tables, figures, statistics), oracle_headroom.py (greedy perfect-information reference)
manuscripts/     LaTeX source of the revised paper;  response/  response letter (LaTeX)
submission/      compiled revised manuscript and point-by-point response (PDF)
results/raw/     one JSON record per (tag, method, seed)
```

## Reproducing

```bash
pip install -r requirements.txt
# one run (1000 episodes; a GPU is recommended)
python -m stag_fedsac.evaluation.run_experiment --method full --seed 0 --episodes 1000 --out results/raw/main_full_s0.json
# methods: ssf llf persist lstm_drl fedddpg full a1_nojoint a2_nosched a3_noemb a4_zone a5_hieravg iso flat central
# the whole grid of the paper
python scripts/run_queue.py jobs.json results/raw --episodes 1000
# tables, figures, statistics
python analysis/aggregate.py
```

No trained checkpoints are distributed.

## License
MIT
