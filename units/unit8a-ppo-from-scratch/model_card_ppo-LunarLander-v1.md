---
library_name: pytorch
tags:
- LunarLander-v2   # tag the course certification app searches for; the agent itself is trained on LunarLander-v3
- LunarLander-v3
- ppo
- deep-reinforcement-learning
- reinforcement-learning
- custom-implementation
- cleanrl
- deep-rl-course
model-index:
- name: PPO
  results:
  - task:
      type: reinforcement-learning
      name: reinforcement-learning
    dataset:
      name: LunarLander-v3
      type: LunarLander-v3
    metrics:
    - type: mean_reward
      value: 239.42 +/- 31.53
      name: mean_reward
      verified: false
---

# PPO agent playing LunarLander-v3

A Proximal Policy Optimization (PPO) agent trained from scratch with my own implementation, following the
[CleanRL](https://docs.cleanrl.dev/rl-algorithms/ppo/) `ppo.py` design, as part of the
[Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course) (Unit 8, PPO).

<video controls autoplay loop muted width="480"
  src="https://huggingface.co/elliemci/ppo-LunarLander-v1/resolve/main/replay.mp4"></video>

## Results

| Evaluation | Episodes | Mean reward |
|---|---|---|
| **Greedy actions (most likely action), reported result** | 100 | **239.42 ± 31.53** |
| Sampled actions (stochastic policy) | 10 | 158.03 ± 22.29 |

- LunarLander is usually considered solved at a mean reward of 200 or more. The mean minus one standard deviation is
  about 208.
- The agent was trained with sampled actions, but evaluating with the most likely action (as Stable-Baselines3's
  `evaluate_policy` does by default) scores much higher, because sampling adds random mistakes.
- The numbers have not been verified by a third party (`verified: false`), and they vary a little with the
  evaluation episodes, so expect a few points either way.

## Training

The weights come from one run of 4,000,000 timesteps (seed 1) on a Colab T4 runtime, at about 2,500 steps per second.

| Setting | Value |
|---|---|
| Environment | `LunarLander-v3` (Gymnasium, discrete actions) |
| Parallel environments | 8 |
| Steps per rollout | 1024 (batch of 8,192) |
| Mini-batches / update epochs | 32 (size 256) / 4 |
| Learning rate | 2.5e-4, linearly annealed to 0 |
| Discount (gamma) / GAE lambda | 0.999 / 0.98 |
| Clip coefficient | 0.2 (clipped value loss) |
| Entropy / value-loss coefficients | 0.01 / 0.5 |
| Max gradient norm | 0.5 |
| Advantage normalisation | on |

The network is a separate actor and critic, each with two hidden layers of 64 units and Tanh activations, using
orthogonal initialisation.

Command that produced the weights:

```bash
python ppo.py --env-id="LunarLander-v3" --total-timesteps=4000000 --num-envs=8 --num-steps=1024 \
  --num-minibatches=32 --update-epochs=4 --gamma=0.999 --gae-lambda=0.98 --ent-coef=0.01
```

TensorBoard logs of the training runs are in the `logs/` folder.

## Usage

`model.pt` is a PyTorch `state_dict` for the actor-critic network below.

```python
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from huggingface_hub import hf_hub_download


class Agent(nn.Module):
    def __init__(self, obs_dim=8, n_actions=4):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, n_actions))


agent = Agent()
path = hf_hub_download("elliemci/ppo-LunarLander-v1", "model.pt")
agent.load_state_dict(torch.load(path, map_location="cpu"))
agent.eval()

env = gym.make("LunarLander-v3")   # needs `pip install "gymnasium[box2d]"`
obs, _ = env.reset(seed=0)
done, total = False, 0.0
while not done:
    with torch.no_grad():
        action = agent.actor(torch.tensor(obs, dtype=torch.float32)).argmax().item()
    obs, reward, terminated, truncated, _ = env.step(action)
    total += reward
    done = terminated or truncated
print("episode return:", total)
```

## Notes and limitations

- Evaluation and the replay video use greedy actions. Sampling from the policy gives lower scores (see Results).
- This is a single training run with a single seed, so the result may not repeat exactly with other seeds.
- The model was trained and evaluated on `LunarLander-v3`. It will not transfer to other environments.
- The training-reward curve is not available for these runs: a bug in my episode-return logging meant only the loss and
  speed metrics were written to the TensorBoard logs. The bug is fixed for later runs.
