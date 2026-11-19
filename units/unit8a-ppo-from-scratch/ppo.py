
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

import argparse                         # use to set common variables for cml interfaces
import os
import random
import time

def strtobool(val):
  """distutils was removed in Python 3.12, so keep a local copy of strtobool."""
  val = str(val).lower()
  if val in ("y", "yes", "t", "true", "on", "1"): return 1
  if val in ("n", "no", "f", "false", "off", "0"): return 0
  raise ValueError(f"invalid truth value {val!r}")

import gymnasium as gym                 # use gymnasium since gym is depricated
import numpy as np

import torch
import torch.nn as nn                   # torch neural network module
import torch.optim as optim             # optimizer module

from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter # tensorboard API to log data for visualization 

from huggingface_hub import HfApi, upload_folder 
from huggingface_hub.repocard import metadata_eval_result, metadata_save

from pathlib import Path
import datetime
import tempfile
import json
import shutil
import imageio

import wandb


from wasabi import Printer
msg = Printer()


def parse_args():
  """ Define and parse command-line arguments of the control hyperparameters and training settings,
      and whether to track with Weights & Biases. Variables are defined with name, type, default
      value and help text for documentation. Variables are stored as attributes of the parser object used 
      to configure and control the training process and pushing to HF Hub. """
  parser = argparse.ArgumentParser()
  parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"), help="the name of this experiment")
  parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
  parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, `torch.backends.cudnn.deterministic=False`") # to eliminate source of variability and reproduce experiments, non-deterministic algorithms are used by default
  parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, cuda will be enabled by default")                           # use cuda whenever posible
  parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, this experiment will be tracked with Weights and Biases")  # track flag to log off expriments with W&B
  parser.add_argument("--wandb-project-name", type=str, default="PPOwithCleanRL", help="the wandb's project name")
  parser.add_argument("--wandb-entity", type=str, default=None, help="the team of wandb's project")                                                                                   # defalt wandb entity is username
  parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="wheather to capture videos of the agent performances (check out `videos` folder)")

  # algorithm specific arguments
  parser.add_argument("--env-id", type=str, default="LunarLander-v3", help="the id of the environment")
  parser.add_argument("--total-timesteps", type=int, default=50000, help="total timesteps of the experiments")
  parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="the learning rate of the optimizer")
  parser.add_argument("--num-envs", type=int, default=4, help="the number of parallel game environments")  
  parser.add_argument("--num-steps", type=int, default=128, help="the number of steps to run in each environment per policy rollout") # number of steps want to collect, for each policy rollout 4*128=512 data points for training
  parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggle learning rate annealing for policy and value networks")
  parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Use GAE for advantage computation") # General Advantage Estimation flag for advantage calculations
  parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma") # discount variable gamma with default value 0.99
  parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation") # lambda variable for gae computation
  parser.add_argument("--num-minibatches", type=int, default=4, help="the number of mini-batches") # for braking the batch into number of mini batches
  parser.add_argument("--update-epochs", type=int, default=4, help="the K epochs to update the policy")
  parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles advantages normalization")
  parser.add_argument("--clip-coef", type=float, default=0.2, help="the surrogate clipping coefficient") # clipped policy coefficent used to calculate PPO policy loss
  parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.") # for value loss clipping
  parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy")
  parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
  parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping") # used for Global Gradient Clipping by setting a max gradient norm with default value of 0.5
  parser.add_argument("--target-kl", type=float, default=None, help="the target KL divergence threshold")

  # continue training from saved weights
  parser.add_argument("--load-model", type=str, default=None, help="HF repo id (user/repo) or local path of a model.pt to initialise the agent from, to continue training")

  # evaluation settings
  parser.add_argument("--greedy-eval", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="evaluate with the most likely action instead of sampling from the policy")
  parser.add_argument("--eval-episodes", type=int, default=100, help="number of episodes used for the final evaluation")

  # adding HuggingFace argument
  parser.add_argument("--repo-id", type=str, default="elliemci/ppo-LunarLander-v1", help="id of the model repository from the Hugging Face Hub")

  args = parser.parse_args()
  args.batch_size = int(args.num_envs * args.num_steps)
  args.minibatch_size = int(args.batch_size // args.num_minibatches)

  return args


def make_env(env_id, seed, idx, capture_video, run_name):
    """Helper function that returns a function which creates a single instance 
       of the specified Gymnasium environment. """
    def thunk():
      """ A nested function, encapsulates creation of single environment for the 
          vectorized environment wrapper gymnasium.vector.SyncVectorEnv 
          and appling wrappers and seeding from make_env. """
      env = gym.make(env_id) 
      # use a RecodEpisodeStatistics gymnasium wrapper to record episotic return
      # the wrapper populated the info variable of env.step(action) with episotic return
      env = gym.wrappers.RecordEpisodeStatistics(env) 
      # flag to capture videos only for the first of the parallel environments
      #if capture_video:
          #if idx == 0:
              #env = gym.wrappers.RecordVideo(env, f"videos/{run_name}") # Use gym prefix
      # set seeds to reproduce
      #env.seed(seed)
      env.action_space.seed(seed)
      env.observation_space.seed(seed)
      return env
      
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """ Function that initalizes the weights and biases of a 
        neural network layer in orthogonal initialization on the 
        layer weight and constant initialization of the layer bias, called
        by Agent class __init__ when neural networks are being built. """

    # Note: the initial weight matrix of a neural network layer to an orthogonal matrix, 
    # An orthogonal matrix is one whose inverse is equal to its transpose 
    # When a vector is multiplied by an orthogonal matrix, the vector's length is preserved 
    # Orthogonal initialization creates a weight matrix where the dot product between any two 
    # distinct rows/ columns is zero, and the dot product of a row/ column with itself is one
    
    torch.nn.init.orthogonal_(layer.weight, std) # ppo uses orthogonal initalization
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):

    def __init__(self, envs):
        super().__init__()
        # Critics network compoed of three layers with hyperbolic Thangent as an activation function
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)), # product of the observation space shape
            nn.Tanh(), 
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),   # std=1, one output feature
        )
        # Actor network
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)), # std=sqrt(2) for orthogonal init
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),    # std=0.01 number of actions output features
        )

    def get_value(self, x):
      """ Implement the critic inference, by passing the observation. """
      return self.critic(x)

    def get_action_and_value(self, x, action=None):
      """ For the actor inference, bundle the results for an observation x 
          with the critic inference"""
      logits = self.actor(x)              # unnormalizaed action probabilities
      probs = Categorical(logits=logits)  # Soft max operation to get the action probability distribution
      if action is None:
          action = probs.sample() # sample actions in the rollout phase
      return action, probs.log_prob(action), probs.entropy(), self.critic(x)

#print(agent)

def package_to_hub(repo_id,
              model,
              hyperparameters,
              eval_env,
              video_fps=30,
              commit_message="Push agent to the Hub",
              token= None,
              logs=None
              ):
  """
  Evaluates the trained model, generates a model card and uploads the model to 
  Hugging Face Hub. Note, the video is generated and uploaded manualy after the
  model is uploaded to the Hub.

  :param repo_id: id of the model repository from the Hugging Face Hub
  :param model: trained model
  :param eval_env: environment used to evaluate the agent
  :param fps: number of fps for rendering the video
  :param commit_message: commit message
  :param logs: directory on local machine of tensorboard logs you'd like to upload
  """
  msg.info(
        "This function will save, evaluate, generate a video of your agent, "
        "create a model card and push everything to the hub. "
        "It might take up to 1min. \n "
        "This is a work in progress: if you encounter a bug, please open an issue."
    )
  # Step 1: Clone or create the repo
  repo_url = HfApi().create_repo(
        repo_id=repo_id,
        token=token,
        private=False,
        exist_ok=True,
    )

  with tempfile.TemporaryDirectory() as tmpdirname:
    tmpdirname = Path(tmpdirname)

    # Step 2: Save the model
    torch.save(model.state_dict(), tmpdirname / "model.pt")

    # Step 3: Evaluate the model and build JSON
    mean_reward, std_reward = _evaluate_agent(eval_env, hyperparameters.eval_episodes, model, hyperparameters.greedy_eval)
    msg.good(f"Evaluation over {hyperparameters.eval_episodes} episodes (greedy={hyperparameters.greedy_eval}): mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # First get datetime
    eval_datetime = datetime.datetime.now()
    eval_form_datetime = eval_datetime.isoformat()

    evaluate_data = {
        "env_id": hyperparameters.env_id,
        "mean_reward": mean_reward,
        "std_reward": std_reward, # Corrected std_reward to std_reward
        "n_evaluation_episodes": hyperparameters.eval_episodes,
          "greedy_eval": hyperparameters.greedy_eval,
        "eval_datetime": eval_form_datetime,
    }

    # Write a JSON file
    with open(tmpdirname / "results.json", "w") as outfile:
      json.dump(evaluate_data, outfile)

    # Step 4: Generate a video
    video_path =  tmpdirname / "replay.mp4"
    record_video(eval_env, model, video_path, video_fps, hyperparameters.greedy_eval)

    # Step 5: Generate the model card
    generated_model_card, metadata = _generate_model_card("PPO", hyperparameters.env_id, mean_reward, std_reward, hyperparameters)
    _save_model_card(tmpdirname, generated_model_card, metadata)

    # Step 6: Add logs if needed
    if logs:
      _add_logdir(tmpdirname, Path(logs))

    msg.info(f"Pushing repo {repo_id} to the Hugging Face Hub")

    repo_url = upload_folder(
            repo_id=repo_id,
            folder_path=tmpdirname,
            path_in_repo="",
            commit_message=commit_message,
            token=token,
        )

    msg.info(f"Model is pushed to the Hub. You can view your model here: {repo_url}")
  return repo_url


def _evaluate_agent(env, n_eval_episodes, policy, greedy=True):
  """
  Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
  :param env: The evaluation environment
  :param n_eval_episodes: Number of episode to evaluate the agent
  :param policy: The agent
  """
  episode_rewards = []
  for episode in range(n_eval_episodes):
    state = env.reset()[0] # Access observation
    step = 0
    done = False
    total_rewards_ep = 0

    while done is False:
      state = torch.Tensor(state).to(device)
      with torch.no_grad():
        if greedy:
          action = policy.actor(state).argmax(dim=-1)   # most likely action
        else:
          action, _, _, _ = policy.get_action_and_value(state)   # sample from the policy
      # step returns (observation, reward, terminated, truncated, info)
      new_state, reward, terminated, truncated, info = env.step(action.cpu().numpy())
      done = terminated or truncated
      total_rewards_ep += reward
      if done:
        break
      state = new_state
    episode_rewards.append(total_rewards_ep)
  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)

  return mean_reward, std_reward


def record_video(env, policy, out_directory, fps=30, greedy=True):
  images = []
  done = False
  state = env.reset()[0] # Access observation
  img = env.render() # Use standard render call
  images.append(img)
  while not done:
    state = torch.Tensor(state).to(device)
    # take the action (index) that have the maximum expected future reward given that state
    with torch.no_grad():
      if greedy:
        action = policy.actor(state).argmax(dim=-1)   # most likely action
      else:
        action, _, _, _ = policy.get_action_and_value(state)
    # step returns (observation, reward, terminated, truncated, info)
    state, reward, terminated, truncated, info = env.step(action.cpu().numpy()) # put next_state = state for recording logic
    done = terminated or truncated
    img = env.render() # Use standard render call
    images.append(img)
  imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)


def _generate_model_card(model_name, env_id, mean_reward, std_reward, hyperparameters):
  """
  Generate the model card for the Hub
  :param model_name: name of the model
  :env_id: name of the environment
  :mean_reward: mean reward of the agent
  :std_reward: standard deviation of the mean reward of the agent
  :hyperparameters: training arguments
  """
  # Step 1: Select the tags
  metadata = generate_metadata(model_name, env_id, mean_reward, std_reward)

  # Transform the hyperparams namespace to string
  converted_dict = vars(hyperparameters)
  converted_str = str(converted_dict)
  converted_str = converted_str.split(", ")
  converted_str = '\n'.join(converted_str)

  # Step 2: Generate the model card
  model_card = f"""
# PPO Agent Playing {env_id}

This is a trained model of a PPO agent playing {env_id}.

Evaluation: {hyperparameters.eval_episodes} episodes with {'greedy (most likely action)' if hyperparameters.greedy_eval else 'sampled'} actions, mean_reward = {mean_reward:.2f} +/- {std_reward:.2f}.

# Hyperparameters
```python
  {converted_str}
  ```
  """
  return model_card, metadata

def generate_metadata(model_name, env_id, mean_reward, std_reward):
  """
  Define the tags for the model card
  :param model_name: name of the model
  :param env_id: name of the environment
  :mean_reward: mean reward of the agent
  :std_reward: standard deviation of the mean reward of the agent
  """
  metadata = {}
  metadata["tags"] = [
        env_id,
        "ppo",
        "deep-reinforcement-learning",
        "reinforcement-learning",
        "custom-implementation",
        "deep-rl-course"
  ]

  # Add metrics
  eval = metadata_eval_result(
      model_pretty_name=model_name,
      task_pretty_name="reinforcement-learning",
      task_id="reinforcement-learning",
      metrics_pretty_name="mean_reward",
      metrics_id="mean_reward",
      metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
      dataset_pretty_name=env_id,
      dataset_id=env_id,
  )

  # Merges both dictionaries
  metadata = {**metadata, **eval}

  return metadata

def _save_model_card(local_path, generated_model_card, metadata):
    """Saves a model card for the repository.
    :param local_path: repository directory
    :param generated_model_card: model card generated by _generate_model_card()
    :param metadata: metadata
    """
    readme_path = local_path / "README.md"
    readme = ""
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
            readme = f.read()
    else:
        readme = generated_model_card

    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme)

    # Save our metrics to Readme metadata
    metadata_save(readme_path, metadata)


def _add_logdir(local_path: Path, logdir: Path):
  """Adds a logdir to the repository.
  :param local_path: repository directory
  :param logdir: logdir directory
  """
  if logdir.exists() and logdir.is_dir():
    # Add the logdir to the repository under new dir called logs
    repo_logdir = local_path / "logs"
    
    # Delete current logs if they exist
    if repo_logdir.exists():
      shutil.rmtree(repo_logdir)

    # Copy logdir into repo logdir
    shutil.copytree(logdir, repo_logdir)


if __name__ == "__main__":
  # parse command-line arguments, default values of not provided varibles are used
  args = parse_args()
  # arguments values can be changed though cml
  # print(args)

  # generate a run name
  run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
  
  # use W&B experiment tracking %python ppo.py --track  goto url to see thr traced experiment
  if args.track:
      
      # init a W&B experiment, logged metrics and visualization on W&B project dashbord
      wandb.init(
          project=args.wandb_project_name,
          entity=args.wandb_entity,
          sync_tensorboard=True,  # W&B syncs with the TensorBoard metric
          config=vars(args),
          name=run_name,
          monitor_gym=True,       # uploads the videos in the W&B
          save_code=True,
      )

  # set up Tensorboard, by default it save logs to runs/, specify run_nam subdir
  writer = SummaryWriter(f"runs/{run_name}")
  # with string formating encode args variables into a Markdown table logged to TensorBoard
  writer.add_text(
      "hyperparameters",
      # for each pair of vargs(args) dictionary creates a string that is a row in a Markdown table
      "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
  )

  # TRY NOT TO MODIFY: seeding
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.backends.cudnn.deterministic = args.torch_deterministic

  # for pytorch a device variable that tries to use a gpu if availbel
  device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

  # setup vectorized environments, gimansium.vector.SyncEnv constructor expects a 
  # list of environment generation functions responsible for creating a single env
  envs = gym.vector.SyncVectorEnv(
      [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)] 
  )
  assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

  #print("envs single obeservation space.shape:", envs.single_observation_space.shape)
  #print("envs single action space size:", envs.single_action_space.n)
  
  # set Adam optimizaer epsilon parameter which default value is 1e-08
  if args.total_timesteps <= 0 and not args.load_model:
      raise SystemExit("--total-timesteps=0 only evaluates and pushes an existing model, so --load-model is required; "
                       "refusing to push an untrained agent to the Hub")
  agent = Agent(envs).to(device)
  if args.load_model:
      model_path = args.load_model
      if not os.path.isfile(model_path):
          from huggingface_hub import hf_hub_download
          model_path = hf_hub_download(repo_id=args.load_model, filename="model.pt")
      agent.load_state_dict(torch.load(model_path, map_location=device))
      msg.good(f"Loaded agent weights from {args.load_model}")
  # creat Adam potimizer wrt the agent parameters, setting up the learning rate and match eps as the ppo original implementation
  optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

  # ALGO Logic: Storage setup for the 4*128=512 data points, batch size, which include variables with shape (num_steps, num_envs) - observations, actions, log probabilities, actions and values
  obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
  actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
  logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
  rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
  dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
  values = torch.zeros((args.num_steps, args.num_envs)).to(device)

  # TRY NOT TO MODIFY: start the game
  global_step = 0            # tracks number of environment steps
  start_time = time.time()   # to calculate frames per second
  reset_output = envs.reset() # Capture the output of reset
  print(f"Output of envs.reset(): {reset_output}") # Print the output
  print(f"Type of envs.reset() output: {type(reset_output)}") # Print the type
  if isinstance(reset_output, tuple):
      print(f"Length of envs.reset() tuple: {len(reset_output)}")
      print(f"Type of observations part: {type(reset_output[0])}")
      if hasattr(reset_output[0], 'shape'):
          print(f"Shape of observations part: {reset_output[0].shape}")

  next_obs = torch.Tensor(reset_output[0]).to(device)    # to store the initial observation
  next_done = torch.zeros(args.num_envs).to(device)      # to store initial termiantion condition
  num_updates = args.total_timesteps // args.batch_size  # number of training interations/updates, 
  
  # 1 update coresponds to 4 itterations: Policy Rollouts -> Rollouts Data -> Policy Training -> Updated Agent -> Policy Rollouts etc.

  # set the training loop based on the number updates
  for update in range(1, num_updates + 1):
      # Annealing the rate with anneal-lr flag setup and fractio variable
      if args.anneal_lr:
          frac = 1.0 - (update - 1.0) / num_updates # 1 for the fist with linear decrease to 0 by the end of the updates
          lrnow = frac * args.learning_rate
          optimizer.param_groups[0]["lr"] = lrnow   # pytorch api to update learning rate

      # policy iteration loop 
      for step in range(0, args.num_steps):
          global_step += 1 * args.num_envs # beacuse each policy step happens at the vector environment
          obs[step] = next_obs             # store in the storage variables
          dones[step] = next_done

          # ALGO LOGIC: action logic, during the Rollout no need to cash gradiant
          with torch.no_grad():
              action, logprob, _, value = agent.get_action_and_value(next_obs)
              values[step] = value.flatten()
          actions[step] = action # in the storage variables
          logprobs[step] = logprob

          # TRY NOT TO MODIFY: execute the game and log data., transfer values to GPU
          next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy()) # Unpack 5 values
          done = np.logical_or(terminated, truncated) # Combine terminated and truncated
          rewards[step] = torch.tensor(reward).to(device).view(-1)
          next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

          # log data: in gymnasium the vector env `info` is a dict of arrays, with the
          # finished episodes flagged in the boolean mask info["_episode"]
          if "episode" in info:
              finished = info.get("_episode", np.ones(args.num_envs, dtype=bool))
              for idx in np.where(finished)[0]:
                  ep_return = float(info["episode"]["r"][idx])
                  ep_length = int(info["episode"]["l"][idx])
                  print(f"global_step={global_step}, episodic_return={ep_return:.2f}")
                  # track metrics with tensorboard
                  writer.add_scalar("charts/episodic_return", ep_return, global_step)
                  writer.add_scalar("charts/episodic_length", ep_length, global_step)

      # modified code from the original PPO implementation, bootstrap value if the epizots are not done not done
      with torch.no_grad():
          next_value = agent.get_value(next_obs).reshape(1, -1)
          if args.gae:
              advantages = torch.zeros_like(rewards).to(device)
              lastgaelam = 0
              for t in reversed(range(args.num_steps)):
                  if t == args.num_steps - 1:
                      nextnonterminal = 1.0 - next_done
                      nextvalues = next_value
                  else:
                      nextnonterminal = 1.0 - dones[t + 1]
                      nextvalues = values[t + 1]
                  delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                  advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
              returns = advantages + values # ppo implementation which is different than the regular implementation returns - values
          else:
              returns = torch.zeros_like(rewards).to(device)
              for t in reversed(range(args.num_steps)):
                  if t == args.num_steps - 1:
                      nextnonterminal = 1.0 - next_done
                      next_return = next_value
                  else:
                      nextnonterminal = 1.0 - dones[t + 1]
                      next_return = returns[t + 1]
                  returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
              advantages = returns - values

      # variables to store the flatten storage variables of the batch
      b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
      b_logprobs = logprobs.reshape(-1)
      b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
      b_advantages = advantages.reshape(-1)
      b_returns = returns.reshape(-1)
      b_values = values.reshape(-1)

      # Optimizing the policy and value network
      b_inds = np.arange(args.batch_size) # for training need all the indices from a batch
      clipfracs = []
      for epoch in range(args.update_epochs):
          np.random.shuffle(b_inds) # shuffle batch indices for each epoch
          # loop throgh the entire batch , one mini batch at a time, where the min batch indices contain 128 of the randomized batch indeces
          for start in range(0, args.batch_size, args.minibatch_size):
              end = start + args.minibatch_size
              mb_inds = b_inds[start:end]
              
              # forward pass on mini batch observation, passing mini batch actions
              _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
              # logarithmic subtraction between the new log probabilities and the old associated with the aactions in the policy roll phase
              logratio = newlogprob - b_logprobs[mb_inds]
              # use the exp to get the ratio from the log ratio; Note during the first pass the ratio is all 1's because there were no changes of the parameters
              ratio = logratio.exp()

              # debug variables KL Divergence to help understant how aggresivly the policy updates
              with torch.no_grad():
                  # calculate approx_kl http://joschu.net/blog/kl-approx.html sugest that the approx_kl is a better estimator than the original implementation (-logratio).mean()
                  old_approx_kl = (-logratio).mean()
                  approx_kl = ((ratio - 1) - logratio).mean()
                  # another debug variable clip fraction variablehelp understand how often the clipping is triggered
                  clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

              # Advantage Normalization
              mb_advantages = b_advantages[mb_inds]
              if args.norm_adv:
                  mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8) # small scalar value to prevent divisio by 0

              # Policy loss
              pg_loss1 = -mb_advantages * ratio
              pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
              pg_loss = torch.max(pg_loss1, pg_loss2).mean() # max of negatives is the same as the paper min of positives

              # Value loss
              newvalue = newvalue.view(-1)
              if args.clip_vloss:   # original PPO implementation
                  v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                  v_clipped = b_values[mb_inds] + torch.clamp(
                      newvalue - b_values[mb_inds],
                      -args.clip_coef,
                      args.clip_coef,
                  )
                  v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                  v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                  v_loss = 0.5 * v_loss_max.mean()
              else:
                  v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean() # loss without clipping
              
              # Entropy loss is included in the over all loss
              entropy_loss = entropy.mean() # measure of chaos in action probability distribution, maximizing entropy, would encurage the aget to explore more 
              # set over all loss by minimizing the policy loss and the value loss and maximizing the entropy loss
              loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

              optimizer.zero_grad()
              loss.backward()
              # set Global Gradient Clipping
              nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
              optimizer.step()

          # Early Stopping at a baatch level: add a condition to beack up if the KL divergence grows aboveover taget_kl trashhod, dafault set to 0.015 by OpenAI Spinning
          if args.target_kl is not None:
              if approx_kl > args.target_kl:
                  break
      
      # another debug variable is the explained variace which tell if the value function is a good indicator of the returns
      y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
      var_y = np.var(y_true)
      explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

      # TRY NOT TO MODIFY: record rewards for plotting purposes
      writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
      writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
      writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
      writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
      writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
      writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
      writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
      writer.add_scalar("losses/explained_variance", explained_var, global_step)
      #print("Step Per Second:", int(global_step / (time.time() - start_time)))
      writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

  envs.close()   # shut down vextorised env subprocesses and release system recources, memory CPU/ GPU times the vectorized environment
  writer.close() # close writer to ensure all pending events are writen to disk, check the url for the recorded metrics visualization

  # evaluate the trained agent, generate the video and model card, and push everything to the HF Hub
  eval_env = gym.make(args.env_id, render_mode='rgb_array')
  package_to_hub(repo_id = args.repo_id,
                 model = agent,
                 hyperparameters = args,
                 eval_env = eval_env,
                 logs = f"runs/{run_name}",
                 )
