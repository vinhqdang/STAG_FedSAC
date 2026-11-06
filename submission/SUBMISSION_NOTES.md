# Revision 1 submission note

- **Journal:** Journal of Network and Systems Management
- **Submission ID:** ef0f267a-28d0-4452-a345-18b26b7cdcb9 (revision of v1.0)
- **Submitted:** 2026-11-06 (reported by the author; requested deadline was 2026-11-18)
- **Repository state at submission:** commit `8cf2186` on `main`

## Files submitted
| File | Content |
|---|---|
| `latex_source.zip` | LaTeX source of the revised manuscript (no PDF inside; builds to 56 pages) |
| `response_to_reviewers.pdf` | Point-by-point response to the editor and Reviewers 1 and 2 (7 pages) |
| `revised_manuscript.pdf` | Compiled manuscript, kept here for reference |

## What the revision contains
- Reviewer 1: roles and data flow, per-AP agent definition, reward terms and weight rationale (with a sensitivity study), history/schedule inputs, scope of the federation and privacy claim.
- Reviewer 2: MAC contention (Bianchi model), channel-switching cost, EDCA airtime mapping, ST-GCAT dimensions.
- Editor: all results regenerated from logged runs (`results/raw/`, `analysis/aggregate.py`); claims revised; limitations paragraph added.

## Points to remember if reviewers ask
- Results are simulation-only; the main comparison uses 3 seeds and 1000 episodes (the first version stated 2000).
- The "Proposed" configuration (no latent embedding in the policy state) was singled out after the ablation results were available; the federation, robustness and reward-weight experiments use the latent-embedding configuration.
- Learned controllers match, but do not clearly exceed, the static plan SSF in reward (0.182 vs 0.181); latency is 11.8% lower. The forecasting gain from the timetable is the clearest result.
- The response letter does not list the problems found in the first submission's code and results (hard-coded result figures, untrained baselines, ablation bugs, unused datasets). The repository history documents the corrected pipeline.

## Next
Wait for the editorial decision. If a further revision is requested, results can be regenerated with `scripts/run_queue.py` and `analysis/aggregate.py`.
