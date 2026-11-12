"""
version9 - Continuous actions: a Gaussian policy (InvertedPendulum, MuJoCo).

What this file teaches
----------------------
Nothing in the policy-gradient estimator assumed discrete actions; we only
need log pi_theta(a | s) and its gradient. For a continuous action space
we use a Gaussian policy

    pi_theta(a | s) = N(a; mu_theta(s), diag(sigma^2)),

    log pi_theta(a | s) = sum_j [ -(a_j - mu_j)^2 / (2 sigma_j^2)
                                  - log sigma_j - 0.5 log(2 pi) ],

where the MLP outputs the mean mu_theta(s) and log sigma is a learned,
state-independent parameter (GaussianMLP in model.py). Exploration now
comes from the noise sigma, which the gradient shrinks as the policy
becomes confident. Greedy evaluation acts with the mean.

Action bounds: the sampled action a is unbounded, so we squash it before
sending it to the environment, a_env = tanh(a) * a_max. The log-probability
is computed for the UNSQUASHED sample a; the squashing is then simply part
of the environment, so the estimator stays correct.

What is new compared to version5 (this file deliberately goes back to the
simple single-environment loop with a mean baseline, to isolate one idea)
--------------------------------
  - GaussianMLP policy and its log-likelihood (summed over action dims).
  - Actions are float vectors (stored as float, never cast to integers).
  - InvertedPendulum-v5, which needs MuJoCo: pip install "gymnasium[mujoco]".

Lecture: https://parag1604.github.io/notes/Reinforcement-Learning/Policy-Gradients

Run:  python version9.py                 (until solved)
      python version9.py --max-iters 3   (quick run)
"""
import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from model import GaussianMLP
from utils import set_seed, plot_rewards, save_model, get_device


def get_trajectory(env, policy, device, argmax=True):
    '''
    Run one episode. argmax=True acts with the mean mu_theta(s);
    argmax=False samples a ~ N(mu_theta(s), sigma^2).
    Returns a list of (s_t, a_t, r_t, done_t) with the UNSQUASHED a_t.
    '''
    a_max = env.action_space.high
    obs, _ = env.reset()
    done = False
    trajectory = []
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        with torch.no_grad():
            mean = policy(obs_tensor).cpu().numpy()
            std = policy.log_std.exp().cpu().numpy()
        if argmax:
            action = mean
        else:
            action = np.random.normal(mean, std)
        action_input = np.tanh(action) * a_max  # squash into the bounds
        n_obs, reward, terminated, truncated, _ = env.step(action_input)
        done = terminated or truncated
        trajectory.append((obs, action, reward, done))
        obs = n_obs
    return trajectory


def evaluate(env, policy, device, num_runs=10):
    total_reward = 0
    with torch.no_grad():
        policy.eval()
        for _ in range(num_runs):
            trajectory = get_trajectory(env, policy, device)
            total_reward += sum([reward for _, _, reward, _ in trajectory])
    return total_reward / num_runs


def train(env, policy, optimizer, device, batch_size=1000, gamma=0.99,
          seed=2024, max_iters=1000, target_reward=980):
    set_seed(seed, env)
    policy.eval()
    eval_rewards = [evaluate(env, policy, device)]
    print(f"Initial Average Reward: {eval_rewards[-1]:.2f}")
    for _ in range(max_iters):
        buffer = []
        while len(buffer) < batch_size:
            buffer.extend(get_trajectory(env, policy, device, argmax=False))

        obs = np.array([x for x, _, _, _ in buffer])
        actions = np.array([x for _, x, _, _ in buffer])
        rewards = np.array([x for _, _, x, _ in buffer])
        dones = np.array([x for _, _, _, x in buffer])
        rtgs = np.zeros_like(rewards)
        for i in range(len(rewards) - 1, -1, -1):
            rtgs[i] = rewards[i] + (0 if dones[i] else gamma * rtgs[i + 1])

        policy.train()
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        acs = torch.tensor(actions, dtype=torch.float32).to(device)
        rtgs = torch.tensor(rtgs, dtype=torch.float32).to(device)

        # NEW: Gaussian log-likelihood, summed over action dimensions
        log_likelihoods = policy.distribution(obs).log_prob(acs).sum(axis=1)
        advantages = (rtgs - rtgs.mean()) / (rtgs.std() + 1e-8)
        loss = -(log_likelihoods * advantages).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        policy.eval()
        eval_rewards.append(evaluate(env, policy, device))
        print(f"Iter {len(eval_rewards)-1} - Avg Reward: {eval_rewards[-1]:.2f}"
              f" - sigma: {policy.log_std.exp().mean().item():.3f}")

        if len(eval_rewards) > 3 and np.mean(eval_rewards[-3:]) > target_reward:
            break
    return eval_rewards


def main(args):
    env = gym.make(args.env_name)
    set_seed(args.seed)

    model = GaussianMLP(
        input_dim=env.observation_space.shape[0],
        hidden_dim=args.hid_dim,
        action_dim=env.action_space.shape[0],
        num_layers=3,
    ).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    eval_rewards = train(env, model, optimizer, args.device,
                         batch_size=args.batch_size, gamma=args.gamma,
                         seed=args.seed, max_iters=args.max_iters)
    print("Saved", plot_rewards(eval_rewards, "version9",
                                batch_size=args.batch_size))
    print("Saved", save_model(model, f"policy_{args.env_name}_version9"))

    if args.render:
        env = gym.make(args.env_name, render_mode="human")
        get_trajectory(env, model, args.device, argmax=True)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--env-name', default='InvertedPendulum-v5')
    parser.add_argument('--hid-dim', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=2000,
                        help='minimum number of transitions per update')
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--max-iters', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--render', action='store_true',
                        help='watch the trained policy for one episode')
    args = parser.parse_args()
    args.device = get_device()
    main(args)
