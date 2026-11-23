# Notebook status

One row per notebook. Hygiene (output/metadata stripping, with the notebooks added) is in commit `1eae918` and verified to leave every cell source
unchanged (`scripts/strip_notebooks.py verify`). Code edits are separate commits, listed in the second column.

Status values: **unchanged** (hygiene only), **smoke-tested** (ran locally with tiny budgets, no errors),
**NEEDS COLAB RUN** (code changed, cannot run here), **NEEDS RETRAIN** (hyperparameters/algorithm changed, new HF push expected).

| notebook | code changed? | runnable locally? | verification | status |
|---|---|---|---|---|
| `unit1-ppo-lunarlander/lunarlander.ipynb` | y: deleted the never-run `VecNormalize.load` continue-training cell | yes | smoke test passed, no cells skipped | smoke-tested |
| `unit1-ppo-lunarlander/lunarlander_ppoipynb.ipynb` | n | yes | smoke test passed with cell 45 skipped (Hub download) | smoke-tested |
| `train-hf-environment/training_Huggy.ipynb` | n | no: needs Unity ML-Agents Huggy binary | none, hygiene only | unchanged |
| `unit2-q-learning/q_learning.ipynb` | y: `record_video` loop `while not (terminated or truncated)` (`3175bb9`) | yes | smoke test passed with cell 85 skipped (downloads another user's model) | smoke-tested |
| `unit3-dqn-atari/atari_invaderts.ipynb` | n | no: Atari-scale training through `rl_zoo3` shell commands | none, hygiene only | unchanged |
| `unit4-reinforce/reinforce_agent.ipynb` | y: `json.dupm` typo to `json.dump` (`41077e2`) and `default=int` for numpy ints (`9870b2b`) | yes | smoke test passed with cell 85 skipped (resumes from a local checkpoint) | smoke-tested |
| `unit5-ml-agents/snowball_pyramid.ipynb` | n | no: Unity ML-Agents | none, hygiene only | unchanged |
| `unit6-a2c-robotics/a2c_robot_training.ipynb` | n | no here: panda-gym needs numpy 2, this Mac's torch needs numpy < 2 (`numpy.long` error) | none, hygiene only | unchanged |
| `unit6-a2c-robotics/tqc_her_pick_and_place.ipynb` (+ `tqc_launch.py`, `configs/tqc_panda_v3.yml`) | new notebook | no: needs a GPU and torch >= 2.8; launcher resume logic is unit-tested, cells parse | **Colab runs done.** 1M steps (100 eval episodes): TQC+HER 36%, -34.08 +/- 21.30 vs A2C 5%, -47.69 +/- 10.21. **2M steps (500 episodes): 44.2%, -30.86 +/- 21.70**, pushed to `elliemci/tqc-PandaPickAndPlace-v3` | **done** for the baseline. Open: card rebuild (`scripts/update_model_cards.py --push`, dry run reviewed); optional `normalize: True` run in its own config/run folder |
| `unit7-multi-agent/ai_vs_ai.ipynb` | n | no: Unity ML-Agents SoccerTwos | none, hygiene only | unchanged |
| `unit5-ml-agents/snowball_target.ipynb` (moved from unit 7) | n | no: Unity ML-Agents | none, hygiene only | unchanged |
| `unit8a-ppo-from-scratch/ppo_implementation.ipynb` + `ppo.py` | n | not as a smoke test: `ppo.py` always pushes to the Hub at the end | ran end to end on a Colab T4 on 2026-11-17 (mean reward 239.4 ± 31.5, greedy, 100 episodes) | unchanged |
| `unit8b-sample-factory-vizdoom/vizDoom.ipynb` | n | no: VizDoom + Sample Factory (Linux) | none, hygiene only | unchanged |
| `offline-rl/decision_transformer.ipynb` | n | no: GPU training on a downloaded dataset | none, hygiene only | unchanged |

## Notes
0. `scripts/sync_from_drive.sh` seeds the repo from Drive and overwrites notebooks, which would undo the strip and code edits. Do not re-run it over edited notebooks; edit in the repo from now on.
1. Smoke tests run on a temporary copy: install and Colab cells are commented out, publish calls (`push_to_hub`,
   `package_to_hub`, `notebook_login`, uploads) are replaced with `pass`, budgets are shrunk, and `HF_HUB_OFFLINE=1`
   with an empty `HF_HOME` is set, so nothing can reach the Hub. The committed notebooks are never modified.
   Run: `venv/bin/python scripts/smoke_test.py units/<unit>/<nb>.ipynb [--skip-cells N,M] [--set name=value]`.
2. Local test environment (Intel Mac, Python 3.12): gymnasium 1.2 for Q-learning (Taxi `is_rainy` needs >= 1.1, and 1.4
   deprecates Taxi-v3), gymnasium 1.0 plus stable-baselines3 2.4 for the SB3 notebooks.

## Needs a Colab run / retrain
None at this point. The two code edits (unit 2, unit 4) were re-run locally.
