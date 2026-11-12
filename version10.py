"""
version10 - Putting it together: Gaussian policy + learned baseline +
parallel environments, on HalfCheetah (MuJoCo).

What this file teaches
----------------------
version9 handled continuous actions with the simplest estimator; version8
had the full machinery (reward-to-go, learned baseline V_phi, vectorised
environments) for discrete actions. This file combines them, which is what
it takes to make progress on a harder locomotion task, HalfCheetah-v5
(1000-step episodes, 6-d actions).

One more practical idea: truncated rollouts. HalfCheetah never terminates,
so every episode is 1000 steps long and its rewards-to-go are long, noisy
sums. With --max-steps K we cut the rollouts of every copy into segments
of K steps and treat each segment end as the end of a trajectory (the
environment itself is NOT reset; the next segment continues from where the
cheetah is). With gamma = 0.95 the effective horizon is about
1 / (1 - gamma) = 20 steps, so little is lost by ignoring rewards more than
K = 150 steps ahead, and we get many more, shorter, trajectories per batch.
(The cut-off makes Q_hat_t slightly too small near the end of each segment;
bootstrapping with V_phi there would fix this, which is the actor-critic
idea.) Evaluation still runs full 1000-step episodes.

What is new compared to version8 and version9
--------------------------------------------
  - version8's loop with version9's GaussianMLP policy (tanh-squashed actions).
  - --max-steps: truncated rollouts.

Lecture: https://parag1604.github.io/notes/Reinforcement-Learning/Policy-Gradients

Run:  python version10.py              (needs pip install "gymnasium[mujoco]")
      python version10.py --max-iters 2 --batch-size 3000 --num-parallel-envs 4
"""
import argparse
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import MLP, GaussianMLP
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
            action = policy(obs_tensor).cpu().numpy()  # the mean
        obs, reward, terminated, truncated, _ = env.step(
            np.tanh(action) * env.action_space.high)
        done = terminated or truncated
        total_reward += reward
    print(f"Reward: {total_reward}")


def get_trajectories(env_vec, policy, device, batch_size=1000, max_steps=0):
    '''
    As in version10, but with Gaussian actions. If max_steps > 0, every
    max_steps steps all partial trajectories are closed (marked done) and
    moved to the buffer: truncated rollouts.
    Returns (s_t, a_t, r_t, done_t) with the UNSQUASHED a_t.
    '''
    a_max = env_vec.single_action_space.high
    obs, _ = env_vec.reset()
    trajectories = [[] for _ in range(env_vec.num_envs)]
    buffer = []
    counter = 0
    while len(buffer) < batch_size:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        with torch.no_grad():
            actions = policy.distribution(obs_tensor).sample().cpu().numpy()
        n_obs, rewards, terminated, truncated, _ = env_vec.step(
            np.tanh(actions) * a_max)
        dones = terminated | truncated
        if max_steps > 0:
            counter = (counter + 1) % max_steps
            if counter == 0:
                dones = np.ones(env_vec.num_envs, dtype=bool)
        for i in range(env_vec.num_envs):
            trajectories[i].append((obs[i].copy(), actions[i], rewards[i], dones[i]))
            if dones[i]:
                buffer.extend(trajectories[i])
                trajectories[i] = []
        obs = n_obs
    return buffer


def evaluate(env_vec, policy, device):
    '''Average return of one full episode (acting with the mean) per copy.'''
    obs, _ = env_vec.reset()
    ended = np.zeros(env_vec.num_envs, dtype=bool)
    total_rewards = np.zeros(env_vec.num_envs)
    while not ended.all():
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        with torch.no_grad():
            actions = policy(obs_tensor).cpu().numpy()  # the mean
        obs, rewards, terminated, truncated, _ = env_vec.step(
            np.tanh(actions) * env_vec.single_action_space.high)
        total_rewards += rewards * (~ended)
        ended |= terminated | truncated
    return total_rewards.mean()


def train(env_vec, policy, value_net, optimizer, value_optimizer, device,
          batch_size=1000, gamma=0.95, seed=2024, max_iters=1000,
          target_reward=750, max_steps=0):
    set_seed(seed, env_vec)
    policy.eval()
    eval_rewards = [evaluate(env_vec, policy, device)]
    print(f"Initial Average Reward: {eval_rewards[-1]:.2f}")
    for _ in range(max_iters):
        start = time.time()
        buffer = get_trajectories(env_vec, policy, device, batch_size,
                                  max_steps)
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
        acs = torch.tensor(actions, dtype=torch.float32).to(device)
        q_vals = torch.tensor(rtgs, dtype=torch.float32).to(device)

        # advantages with the learned baseline (as in version7)
        targets = (q_vals - q_vals.mean()) / (q_vals.std() + 1e-8)
        vals = value_net(obs).squeeze(-1)
        advantages = targets - vals.detach()
        advantages = \
            (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # update policy single step (Gaussian log-likelihood, as in version9)
        log_likelihoods = policy.distribution(obs).log_prob(acs).sum(axis=1)
        loss = -(log_likelihoods * advantages).mean()

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

        if len(eval_rewards) > 3 and np.mean(eval_rewards[-3:]) > target_reward:
            break
    return eval_rewards


def main(args):
    env_vec = make_vec_env(args.env_name, args.num_parallel_envs)
    set_seed(args.seed)

    obs_dim = env_vec.single_observation_space.shape[0]
    action_dim = env_vec.single_action_space.shape[0]
    model = GaussianMLP(obs_dim, args.hid_dim, action_dim,
                        num_layers=args.num_layers).to(args.device)
    value_net = MLP(obs_dim, args.hid_dim, 1,
                    num_layers=args.num_layers).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=args.lr)

    eval_rewards = train(
        env_vec, model, value_net, optimizer, value_optimizer, args.device,
        batch_size=args.batch_size, gamma=args.gamma, seed=args.seed,
        max_iters=args.max_iters, max_steps=args.max_steps)
    env_vec.close()
    print("Saved", plot_rewards(eval_rewards, "version10",
                                batch_size=args.batch_size))
    print("Saved", save_model(model, f"policy_{args.env_name}_version10"))

    if args.render:
        env = gym.make(args.env_name, render_mode="human")
        render_trajectory(env, model, args.device)
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--env-name', default='HalfCheetah-v5')
    parser.add_argument('--hid-dim', type=int, default=32)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=48_000,
                        help='minimum number of transitions per update')
    parser.add_argument('--num-parallel-envs', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=150,
                        help='truncate training rollouts every K steps (0 = off)')
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--max-iters', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--render', action='store_true',
                        help='watch the trained policy for one episode')
    args = parser.parse_args()
    args.device = get_device()
    s = time.time()
    main(args)
    print(f"Total run-time: {time.time() - s:.2f} seconds")
