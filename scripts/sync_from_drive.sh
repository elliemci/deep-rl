#!/usr/bin/env bash
# Copy the curated course files from the Google Drive working folder into this repo.
# Usage: DRIVE_DIR="/path/to/My Drive/ColabNotebooks/DeepRL" scripts/sync_from_drive.sh
# Drive stays the Colab source of truth; the repo is a clean, git-friendly copy.
# Only notebooks, scripts, yml configs and small json configs are copied; weights, logs,
# videos and vendored clones are left behind (see .gitignore).
set -euo pipefail
SRC="${DRIVE_DIR:?set DRIVE_DIR to the Google Drive DeepRL folder}"
DST="$(cd "$(dirname "$0")/.." && pwd)/units"

cp_to() { # cp_to <dest-unit> <src-file-relative-to-SRC>...
  local unit="$1"; shift
  mkdir -p "$DST/$unit"
  for f in "$@"; do cp -p "$SRC/$f" "$DST/$unit/"; done
}

cp_to unit1-ppo-lunarlander               intro_deep_q_learning/lunarlander.ipynb lunarlander_ppoipynb.ipynb
cp_to train-hf-environment  training_Huggy.ipynb
cp_to unit2-q-learning                    q_learning/q_learning.ipynb
cp_to unit3-dqn-atari                     deep_q_learning/atari_invaderts.ipynb
mkdir -p "$DST/unit3-dqn-atari/configs"
for f in dqn.yml dqn_0.yml best_params.yml; do cp -p "$SRC/deep_q_learning/$f" "$DST/unit3-dqn-atari/configs/"; done
cp_to unit4-reinforce                     policy_gradient/reinforce_agent.ipynb
cp_to unit5-ml-agents                     unity_ml_agents/snowball_pyramid.ipynb
cp_to unit6-a2c-robotics                  actor_critics/a2c_robot_training.ipynb
cp_to unit7-multi-agent                   marl_ml_agents/ai_vs_ai.ipynb marl_ml_agents/snowball_target.ipynb
cp_to unit8a-ppo-from-scratch             ppo_CleanRL/ppo.py ppo_CleanRL/ppo_implementation.ipynb ppo_CleanRL/model_card_ppo-LunarLander-v1.md
cp_to unit8b-sample-factory-vizdoom       async_ppo_SampleFactory/vizDoom.ipynb
for e in doom_health_gathering_supreme_experiment deadly_corridor_experiment; do
  mkdir -p "$DST/unit8b-sample-factory-vizdoom/train_dir/$e"
  cp -p "$SRC/async_ppo_SampleFactory/train_dir/$e/config.json" "$DST/unit8b-sample-factory-vizdoom/train_dir/$e/"
done
cp_to offline-rl                          offline_rl/decision_transformer.ipynb
echo "synced into $DST"
