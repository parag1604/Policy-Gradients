"""
version8 - Faster data collection with parallel (vectorised) environments.

What this file teaches
----------------------
The algorithm is exactly version7 (reward-to-go + learned baseline V_phi).
What changes is HOW we collect the batch. Policy gradients are on-policy,
so every update needs a fresh batch, and collecting 40 000 transitions one
environment step at a time is the bottleneck: each step calls the network
on a single observation.

A vector environment steps `num_envs` copies of the environment together.
The policy then runs on a (num_envs, obs_dim) batch in one forward pass.
Each copy restarts automatically when its episode ends, so we keep one
partial trajectory per copy and move it to the buffer once it is complete.
(Parag's runs: about 3500 s for version7 and about 750 s for version8.)

What is new compared to version7
--------------------------------
  - gymnasium.make_vec(...) with `num_envs` copies (SAME_STEP autoreset:
    the observation returned after a terminal step is already the first
    observation of the next episode).
  - get_trajectories(): batched sampling that only keeps complete episodes.
  - evaluate(): one greedy episode in every copy, averaged.
  - The stopping rule averages the last 100 // num_envs evaluations.

Lecture: https://parag1604.github.io/notes/Reinforcement-Learning/Policy-Gradients

Run:  python version8.py                                  (LunarLander-v3)
      python version8.py --max-iters 2 --batch-size 2000  (quick run)
"""
import argparse
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import MLP
from utils import set_seed, plot_rewards, save_model, get_device


def make_vec_env(env_name, num_envs):
    return gym.make_vec(
        env_name, num_envs=num_envs, vectorization_mode='sync',
        vector_kwargs={'autoreset_mode': gym.vector.AutoresetMode.SAME_STEP})


def render_trajectory(env, policy, device):
    '''Run one greedy episode in a single (rendered) environment.'''
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        with torch.no_grad():
            action = torch.argmax(policy(obs_tensor)).item()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f"Reward: {total_reward}")


def get_trajectories(env_vec, policy, device, batch_size=1000):
    '''
    Sample actions a ~ pi_theta(. | s) in all copies at once until the buffer
    holds at least batch_size transitions from COMPLETE episodes.
    Returns a list of (s_t, a_t, r_t, done_t), one episode after another.
    '''
    obs, _ = env_vec.reset()
    trajectories = [[] for _ in range(env_vec.num_envs)]
    buffer = []
    while len(buffer) < batch_size:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        with torch.no_grad():
            actions = torch.distributions.Categorical(
                logits=policy(obs_tensor)).sample().cpu().numpy()
        n_obs, rewards, terminated, truncated, _ = env_vec.step(actions)
        dones = terminated | truncated
        for i in range(env_vec.num_envs):
            trajectories[i].append((obs[i].copy(), actions[i], rewards[i], dones[i]))
            if dones[i]:
                buffer.extend(trajectories[i])
                trajectories[i] = []
        obs = n_obs
    return buffer


def evaluate(env_vec, policy, device):
    '''Average return of one greedy episode in every copy.'''
    obs, _ = env_vec.reset()
    ended = np.zeros(env_vec.num_envs, dtype=bool)
    total_rewards = np.zeros(env_vec.num_envs)
    while not ended.all():
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        with torch.no_grad():
            actions = torch.argmax(policy(obs_tensor), dim=1).cpu().numpy()
        obs, rewards, terminated, truncated, _ = env_vec.step(actions)
        total_rewards += rewards * (~ended)
        ended |= terminated | truncated
    return total_rewards.mean()


def train(env_vec, policy, value_net, optimizer, value_optimizer, device,
          batch_size=1000, gamma=0.99, seed=2024, max_iters=1000,
          target_reward=200):
    set_seed(seed, env_vec)
    window = max(1, 100 // env_vec.num_envs)
    policy.eval()
    eval_rewards = [evaluate(env_vec, policy, device)]
    print(f"Initial Average Reward: {eval_rewards[-1]:.2f}")
    for _ in range(max_iters):
        # NEW: collect data from all copies in parallel
        start = time.time()
        buffer = get_trajectories(env_vec, policy, device, batch_size)
        end = time.time()

        # prepare data
        obs = np.array([x for x, _, _, _ in buffer])
        actions = np.array([x for _, x, _, _ in buffer])
        rewards = np.array([x for _, _, x, _ in buffer])
        dones = np.array([x for _, _, _, x in buffer])
        # discounted rewards-to-go (episodes are contiguous in the buffer)
        rtgs = np.zeros_like(rewards)
        for i in range(len(rewards) - 1, -1, -1):
            rtgs[i] = rewards[i] + (0 if dones[i] else gamma * rtgs[i + 1])

        policy.train()
        value_net.train()
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        acs = torch.tensor(actions, dtype=torch.long).to(device)
        q_vals = torch.tensor(rtgs, dtype=torch.float32).to(device)

        # advantages with the learned baseline (as in version7)
        targets = (q_vals - q_vals.mean()) / (q_vals.std() + 1e-8)
        vals = value_net(obs).squeeze(-1)
        advantages = targets - vals.detach()
        advantages = \
            (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # update policy single step
        ac_logits = policy(obs)
        ac_log_likelihoods = torch.log_softmax(ac_logits, axis=1)
        selected_log_likelihoods = \
            torch.gather(ac_log_likelihoods, 1, acs.view(-1, 1)).squeeze()
        loss = -(selected_log_likelihoods * advantages).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        # update the value network
        val_loss = F.mse_loss(vals, targets)
        value_optimizer.zero_grad()
        val_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
        value_optimizer.step()

        policy.eval()
        value_net.eval()
        eval_rewards.append(evaluate(env_vec, policy, device))
        print(f"Iter {len(eval_rewards)-1} - Avg Reward: {eval_rewards[-1]:.2f}"
              f" - Collected {len(buffer)} samples in {end - start:.2f} seconds")

        if len(eval_rewards) > window and \
                np.mean(eval_rewards[-window:]) > target_reward:
            break
    return eval_rewards


def main(args):
    env_vec = make_vec_env(args.env_name, args.num_parallel_envs)
    set_seed(args.seed)

    obs_dim = env_vec.single_observation_space.shape[0]
    n_actions = env_vec.single_action_space.n
    model = MLP(obs_dim, args.hid_dim, n_actions,
                num_layers=args.num_layers).to(args.device)
    value_net = MLP(obs_dim, args.hid_dim, 1,
                    num_layers=args.num_layers).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=args.lr)

    eval_rewards = train(
        env_vec, model, value_net, optimizer, value_optimizer, args.device,
        batch_size=args.batch_size, gamma=args.gamma, seed=args.seed,
        max_iters=args.max_iters)
    env_vec.close()
    print("Saved", plot_rewards(eval_rewards, "version8",
                                batch_size=args.batch_size))
    print("Saved", save_model(model, f"policy_{args.env_name}_version8"))

    if args.render:
        env = gym.make(args.env_name, render_mode="human")
        render_trajectory(env, model, args.device)
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--env-name', default='LunarLander-v3')
    parser.add_argument('--hid-dim', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=40_000,
                        help='minimum number of transitions per update')
    parser.add_argument('--num-parallel-envs', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--max-iters', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--render', action='store_true',
                        help='watch the trained policy for one episode')
    args = parser.parse_args()
    args.device = get_device()
    s = time.time()
    main(args)
    print(f"Total run-time: {time.time() - s:.2f} seconds")
