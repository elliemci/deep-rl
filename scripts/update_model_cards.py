#!/usr/bin/env python
"""Update the README (model card) of one of my Hugging Face models. DRY RUN BY DEFAULT.

    python scripts/update_model_cards.py tqc-PandaPickAndPlace-v3            # print diff + files, change nothing
    python scripts/update_model_cards.py tqc-PandaPickAndPlace-v3 --push     # really upload (needs a write token)

Only models listed in CARDS are supported. The current card is read from the public repo (no token needed for a
dry run). Existing YAML tags/metadata are kept; the `model-index` metrics are replaced by the values below.
`--push` uses the token from `huggingface-cli login` / HF_TOKEN and uploads, in ONE commit: README.md and the
extra files listed per card. It changes a public page, so look at the dry run first.
"""
import argparse
import difflib
import io
import re
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
OWNER = "elliemci"
GITHUB = "https://github.com/elliemci/deep-rl"

TQC_BODY = """
# **TQC + HER** agent playing **PandaPickAndPlace-v3**

![replay](replay.gif)

A [TQC](https://arxiv.org/abs/2005.04269) agent with [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) trained with
[SB3-Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) through the
[RL-Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) on the sparse-reward robot task
[`PandaPickAndPlace-v3`](https://panda-gym.readthedocs.io/) (panda-gym 3, gymnasium).
It is a "beyond the course" follow-up to Unit 6 of the Hugging Face Deep RL Course, where an A2C agent never learned this task.
**The task is not solved:** the agent succeeds in about a third of the episodes.

## Results

100 evaluation episodes, deterministic actions, same evaluation seed for both agents, after 1M training steps (seed 0).
Success is `info["is_success"]` on the last step of the episode.

| agent | training | success rate | mean reward +/- std | mean episode length |
|---|---|---|---|---|
| [A2C (Unit 6)](https://huggingface.co/elliemci/a2c-PandaPickAndPlace-v3) | on-policy, VecNormalize | 5% | -47.69 +/- 10.21 | 47.7 |
| **TQC + HER (this model)** | off-policy, HER future x4 | **36%** | **-34.08 +/- 21.30** | 34.4 |

One seed and 100 episodes: roughly a +/- 9 point 95% interval on the 36%. The A2C 5% is probably episodes where the target spawns next to the cube
(not verified).

![evaluation success during training](eval_curve.png)

Success rate over 20 evaluation episodes every 50k steps (noisy). The last two points (950k, 1M) come from a run resumed from a checkpoint.

## Training

* Hyperparameters: RL-Zoo `tqc.yml` "her-defaults" (the entry used by `PandaPickAndPlace-v1` there), registered unchanged under the v3 id:
  `MultiInputPolicy`, `HerReplayBuffer(goal_selection_strategy="future", n_sampled_goal=4)`, buffer 1M, batch 2048, gamma 0.95, lr 1e-3, tau 0.05,
  net `[512, 512, 512]`, 2 critics, 1M steps, no observation normalization.
* Versions: rl_zoo3 2.9.2, sb3-contrib 2.9.2, stable-baselines3 2.9.2, panda-gym 3.0.7, gymnasium 1.0.0.
* Code, config and the Colab notebook: [{github}/tree/main/units/unit6-a2c-robotics]({github}/tree/main/units/unit6-a2c-robotics).

## Usage

```python
import gymnasium as gym, panda_gym
from huggingface_sb3 import load_from_hub
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv

path = load_from_hub("elliemci/tqc-PandaPickAndPlace-v3", "tqc-PandaPickAndPlace-v3.zip")
env = DummyVecEnv([lambda: gym.make("PandaPickAndPlace-v3")])
model = TQC.load(path, env=env)            # HerReplayBuffer needs an env at load time

obs = env.reset()
for _ in range(50):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
```
""".replace("{github}", GITHUB)

CARDS = {
    "tqc-PandaPickAndPlace-v3": dict(
        body=TQC_BODY,
        extra_tags=["her", "tqc", "panda-gym"],
        model_name="TQC",
        env="PandaPickAndPlace-v3",
        metrics=[("mean_reward", "-34.08 +/- 21.30"), ("success_rate", "36.00")],
        files={"eval_curve.png": ROOT / "assets/plots/tqc_eval_curve.png",
               "replay.gif": ROOT / "assets/gifs/tqc-PandaPickAndPlace-v3.gif"},
    ),
}


def split_card(text):
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    return (yaml.safe_load(m.group(1)), m.group(2)) if m else ({}, text)


def build(repo_id, spec, old_text):
    meta, _ = split_card(old_text)
    meta["tags"] = list(dict.fromkeys(list(meta.get("tags", [])) + spec["extra_tags"]))
    meta["model-index"] = [{
        "name": spec["model_name"],
        "results": [{
            "task": {"type": "reinforcement-learning", "name": "reinforcement-learning"},
            "dataset": {"name": spec["env"], "type": spec["env"]},
            "metrics": [{"type": t, "value": v, "name": t, "verified": False} for t, v in spec["metrics"]],
        }],
    }]
    front = yaml.safe_dump(meta, sort_keys=False, default_flow_style=False).strip()
    return f"---\n{front}\n---\n{spec['body']}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(CARDS))
    ap.add_argument("--push", action="store_true", help="really upload (default: dry run)")
    a = ap.parse_args()
    spec, repo_id = CARDS[a.model], f"{OWNER}/{a.model}"

    old = Path(hf_hub_download(repo_id, "README.md")).read_text()
    new = build(repo_id, spec, old)
    for name, path in spec["files"].items():
        if not path.exists():
            sys.exit(f"missing local file for upload: {path}")

    print(f"=== {repo_id}: README.md diff (- current, + new) ===")
    sys.stdout.writelines(difflib.unified_diff(old.splitlines(True), new.splitlines(True), "README.md (hub)", "README.md (new)", n=1))
    print("\n=== files that would be uploaded in the same commit ===")
    for name, path in spec["files"].items():
        print(f"  {name:16s} <- {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")
    if not a.push:
        print("\nDRY RUN: nothing was changed. Re-run with --push to upload.")
        return
    from huggingface_hub import CommitOperationAdd
    ops = [CommitOperationAdd("README.md", io.BytesIO(new.encode()))]
    ops += [CommitOperationAdd(n, str(p)) for n, p in spec["files"].items()]
    info = HfApi().create_commit(repo_id, ops, commit_message="Model card: results vs A2C, learning curve, usage, links")
    print("pushed:", info.commit_url)


if __name__ == "__main__":
    main()
