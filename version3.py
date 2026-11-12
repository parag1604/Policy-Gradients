"""
version3 - REINFORCE with a constant baseline.

What this file teaches
----------------------
REINFORCE is unbiased but has very high variance. On CartPole every reward
is +1, so every return is positive and EVERY sampled action gets pushed up;
only the small differences between returns carry the useful signal.
Subtracting a baseline b from the return fixes this:

    grad J ~ (1/N) sum_i ( sum_t grad log pi(a_it | s_it) ) ( R(tau_i) - b )

Trajectories better than b are reinforced, worse ones are suppressed. For
any constant b the expected gradient is unchanged (E[grad log p_theta] = 0),
so only the variance goes down.

What is new compared to version2
--------------------------------
  - One line: b = average return of the batch, and the weight becomes
    R(tau) - b. (Using the same batch for b adds a negligible O(1/N) bias.)

Lecture: https://parag1604.github.io/notes/Reinforcement-Learning/Policy-Gradients
(sections "The variance problem" and "Baselines").

Run:  python version3.py [--max-iters 5]
"""
import argparse
import time

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from model import MLP
from utils import set_seed, plot_rewards, save_model, get_device


def get_trajectory(env, policy, device, argmax=True, render=False):
    '''
    Run one episode. argmax=True acts greedily (for evaluation);
    argmax=False samples a ~ pi_theta(. | s) (for training).
    Returns a list of (s_t, a_t, r_t, done_t).
    '''
    obs, _ = env.reset()
    done = False
    trajectory = []
    while not done:
        if render:
            time.sleep(0.005)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        with torch.no_grad():
            action_logits = policy(obs_tensor)
            if argmax:
                action = torch.argmax(action_logits).item()
            else:
                action = torch.distributions.Categorical(
                    logits=action_logits).sample().item()
        n_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        trajectory.append((obs, action, reward, done))
        obs = n_obs
    return trajectory


def evaluate(env, policy, device, num_runs=10):
    '''Average undiscounted return of the greedy policy.'''
    total_reward = 0
    with torch.no_grad():
        policy.eval()
        for _ in range(num_runs):
            trajectory = get_trajectory(env, policy, device)
            total_reward += sum([reward for _, _, reward, _ in trajectory])
    return total_reward / num_runs


def train(env, policy, optimizer, device, batch_size=1000, seed=2024,
          max_iters=1000, target_reward=195):
    set_seed(seed, env)
    policy.eval()
    eval_rewards = [evaluate(env, policy, device)]
    print(f"Initial Average Reward: {eval_rewards[-1]:.2f}")
    for _ in range(max_iters):
        # collect data: complete trajectories, at least batch_size steps
        buffer = []
        while len(buffer) < batch_size:
            trajectory = get_trajectory(env, policy, device, argmax=False)
            buffer.extend(trajectory)

        # prepare data
        obs = np.array([x for x, _, _, _ in buffer])
        actions = np.array([x for _, x, _, _ in buffer])
        rewards = np.array([x for _, _, x, _ in buffer])
        dones = np.array([x for _, _, _, x in buffer])
        # every step gets the total return R(tau) of its own trajectory
        returns = np.zeros_like(rewards)
        idxs = np.array([0] + (np.where(dones == 1)[0] + 1).tolist())
        for i in range(len(idxs) - 1):
            start = idxs[i]
            end = idxs[i + 1]
            returns[start:end] = rewards[start:end].sum()

        policy.train()
        X_batch = torch.tensor(obs, dtype=torch.float32).to(device)
        y_batch = torch.tensor(actions, dtype=torch.long).to(device)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        # one policy-gradient step on the weighted log-likelihood
        outs = policy(X_batch)
        log_likelihoods = torch.log_softmax(outs, axis=1)
        selected_log_likelihoods = \
            torch.gather(log_likelihoods, 1, y_batch.view(-1, 1)).squeeze()
        # NEW: subtract the average return (a constant baseline)
        baseline = returns.mean()
        advantages = returns - baseline
        scaled_log_likelihoods = selected_log_likelihoods * advantages
        loss = -scaled_log_likelihoods.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        policy.eval()
        eval_rewards.append(evaluate(env, policy, device))
        print(f"Iter {len(eval_rewards)-1} - Avg Reward: {eval_rewards[-1]:.2f}")

        # stop when the last 10 evaluations average above the target
        if len(eval_rewards) > 10 and np.mean(eval_rewards[-10:]) > target_reward:
            break
    return eval_rewards


def main(args):
    env = gym.make(args.env_name)
    set_seed(args.seed)

    model = MLP(
        input_dim=env.observation_space.shape[0],
        hidden_dim=args.hid_dim,
        output_dim=env.action_space.n,
    ).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    eval_rewards = train(env, model, optimizer, args.device,
                         batch_size=args.batch_size, seed=args.seed,
                         max_iters=args.max_iters)
    print("Saved", plot_rewards(eval_rewards, "version3",
                                batch_size=args.batch_size))
    print("Saved", save_model(model, f"policy_{args.env_name}_version3"))

    if args.render:
        env = gym.make(args.env_name, render_mode="human")
        get_trajectory(env, model, args.device, argmax=True, render=True)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--env-name', default='CartPole-v0')
    parser.add_argument('--hid-dim', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='minimum number of transitions per update')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--max-iters', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--render', action='store_true',
                        help='watch the trained policy for one episode')
    args = parser.parse_args()
    args.device = get_device()
    main(args)
