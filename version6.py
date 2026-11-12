"""
version6 - A recurrent policy over a short history of observations.

What this file teaches
----------------------
So far the policy saw only the current observation, pi_theta(a | s_t). That
is enough when the observation is a Markov state. When it is not (partial
observability, e.g. a camera image without velocities), the policy needs
memory. The simplest memory: feed the last `hist` observations
(o_{t-hist+1}, ..., o_t) to an LSTM and act on its final hidden state,
pi_theta(a | o_{t-hist+1:t}). Nothing in the policy-gradient estimator
changes: only the input of the network.

CartPole's 4-d observation (position, velocity, angle, angular velocity)
IS Markov, so by default the history is not needed and this mostly shows
the plumbing. Run with --hide-velocity to drop the two velocities: the
single-frame problem becomes partially observed, and the history is what
lets the policy recover the velocities (by differencing frames).

What is new compared to version5
--------------------------------
  - The RNN policy from model.py and a sliding window of `hist` frames.
  - The optional --hide-velocity wrapper (a partially observed CartPole).
  - Parag's settings for this run: batch 500, gamma 0.95, lr 1e-3, and the
    run stops as soon as one evaluation exceeds 190.

Lecture: https://parag1604.github.io/notes/Reinforcement-Learning/Introduction-to-Reinforcement-Learning
(section "Partially observed MDPs"); estimator as in
https://parag1604.github.io/notes/Reinforcement-Learning/Policy-Gradients

Run:  python version6.py [--max-iters 5] [--hide-velocity]
"""
import argparse
import time

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from model import RNN
from utils import set_seed, plot_rewards, save_model, get_device


def make_env(env_name, hide_velocity=False, render_mode=None):
    '''CartPole, optionally with the two velocity components removed.'''
    env = gym.make(env_name, render_mode=render_mode)
    if hide_velocity:
        keep = [0, 2]  # cart position, pole angle
        space = gym.spaces.Box(env.observation_space.low[keep],
                               env.observation_space.high[keep],
                               dtype=np.float32)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: obs[keep], space)
    return env


def get_trajectory(env, policy, device, argmax=True, render=False, hist=3):
    '''
    Like version6, but the policy input is a (hist, obs_dim) window of the
    most recent observations. At the start, the window is filled with
    copies of the first observation.
    '''
    obs, _ = env.reset()
    done = False
    trajectory = []
    obs = np.stack([obs] * hist)
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
        # store a COPY: the window is shifted in place below
        trajectory.append((obs.copy(), action, reward, done))
        obs[:-1] = obs[1:]
        obs[-1] = n_obs
    return trajectory


def evaluate(env, policy, device, num_runs=10, hist=3):
    '''Average undiscounted return of the greedy policy.'''
    total_reward = 0
    with torch.no_grad():
        policy.eval()
        for _ in range(num_runs):
            trajectory = get_trajectory(env, policy, device, hist=hist)
            total_reward += sum([reward for _, _, reward, _ in trajectory])
    return total_reward / num_runs


def train(env, policy, optimizer, device, batch_size=1000, gamma=0.99, seed=2024,
          max_iters=1000, target_reward=190, hist=3):
    set_seed(seed, env)
    policy.eval()
    eval_rewards = [evaluate(env, policy, device, hist=hist)]
    print(f"Initial Average Reward: {eval_rewards[-1]:.2f}")
    for _ in range(max_iters):
        # collect data: complete trajectories, at least batch_size steps
        buffer = []
        while len(buffer) < batch_size:
            trajectory = get_trajectory(
                env, policy, device, argmax=False, hist=hist)
            buffer.extend(trajectory)

        # prepare data
        obs = np.array([x for x, _, _, _ in buffer])
        actions = np.array([x for _, x, _, _ in buffer])
        rewards = np.array([x for _, _, x, _ in buffer])
        dones = np.array([x for _, _, _, x in buffer])
        # discounted rewards-to-go, Q_hat_t = r_t + gamma * Q_hat_{t+1},
        # computed backwards; the recursion restarts at every episode end
        rtgs = np.zeros_like(rewards)
        for i in range(len(rewards) - 1, -1, -1):
            rtgs[i] = rewards[i] + (0 if dones[i] else gamma * rtgs[i + 1])

        policy.train()
        X_batch = torch.tensor(obs, dtype=torch.float32).to(device)
        y_batch = torch.tensor(actions, dtype=torch.long).to(device)
        rtgs = torch.tensor(rtgs, dtype=torch.float32).to(device)

        # one policy-gradient step on the weighted log-likelihood
        outs = policy(X_batch)
        log_likelihoods = torch.log_softmax(outs, axis=1)
        selected_log_likelihoods = \
            torch.gather(log_likelihoods, 1, y_batch.view(-1, 1)).squeeze()
        # baseline (mean) and normalisation (std) of the rewards-to-go
        advantages = (rtgs - rtgs.mean()) / (rtgs.std() + 1e-8)
        scaled_log_likelihoods = selected_log_likelihoods * advantages
        loss = -scaled_log_likelihoods.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        policy.eval()
        eval_rewards.append(evaluate(env, policy, device, hist=hist))
        print(f"Iter {len(eval_rewards)-1} - Avg Reward: {eval_rewards[-1]:.2f}")

        # stop as soon as one evaluation exceeds the target
        if len(eval_rewards) > 10 and eval_rewards[-1] > target_reward:
            break
    return eval_rewards


def main(args):
    env = make_env(args.env_name, args.hide_velocity)
    set_seed(args.seed)

    model = RNN(
        input_dim=env.observation_space.shape[0],
        hidden_dim=args.hid_dim,
        output_dim=env.action_space.n,
    ).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    eval_rewards = train(env, model, optimizer, args.device,
                         batch_size=args.batch_size, gamma=args.gamma,
                         seed=args.seed,
                         max_iters=args.max_iters, hist=args.hist)
    print("Saved", plot_rewards(eval_rewards, "version6",
                                batch_size=args.batch_size))
    print("Saved", save_model(model, f"policy_{args.env_name}_version6"))

    if args.render:
        env = make_env(args.env_name, args.hide_velocity, render_mode="human")
        get_trajectory(env, model, args.device, argmax=True, render=True,
                       hist=args.hist)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--env-name', default='CartPole-v1')
    parser.add_argument('--hid-dim', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=500,
                        help='minimum number of transitions per update')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--hist', type=int, default=3,
                        help='number of recent observations fed to the LSTM')
    parser.add_argument('--hide-velocity', action='store_true',
                        help='drop the velocities (partially observed CartPole)')
    parser.add_argument('--max-iters', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--render', action='store_true',
                        help='watch the trained policy for one episode')
    args = parser.parse_args()
    args.device = get_device()
    main(args)
