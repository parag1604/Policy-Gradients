"""
version7 - A learned, state-dependent baseline b(s_t) ~ V^pi(s_t) (LunarLander).

What this file teaches
----------------------
The reward-to-go Q_hat_t shrinks along a trajectory, so one constant
baseline (version5) cannot centre it at every step. A state-dependent
baseline b(s_t) is still unbiased (it does not depend on a_t), and the best
choice is close to the value function V^pi(s_t). The weight then becomes an
estimate of the ADVANTAGE

    A_hat_t = Q_hat_t - V_phi(s_t)  ~  A^pi(s_t, a_t) = Q^pi - V^pi,

i.e. "how much better was this action than the policy's average action".

V_phi is a second MLP trained by regression on the rewards-to-go,
    min_phi  mean_t ( V_phi(s_t) - Q_hat_t )^2,
one gradient step per batch, alongside the policy step. It is a Monte-Carlo
critic: it does not bootstrap (no r + gamma V(s')). Bootstrapping is the
step to actor-critic methods (see the Actor-Critic repository).

Scaling: the rewards-to-go are normalised per batch (mean 0, std 1) and
V_phi learns these normalised targets, so the advantage is
normalised Q_hat_t minus V_phi(s_t); the advantages are then normalised
once more, as in version5.

What is new compared to version5
--------------------------------
  - A value network V_phi(s) and its regression loss.
  - Advantages A_hat_t = Q_hat_t - V_phi(s_t) instead of Q_hat_t - mean.
  - A harder task: LunarLander-v3 (needs `pip install "gymnasium[box2d]"`),
    deeper networks (3 layers) and big batches (40 000 transitions).

Lecture: https://parag1604.github.io/notes/Reinforcement-Learning/Policy-Gradients
(section "Baselines with reward-to-go": state-dependent baselines).

Run:  python version7.py                                (Parag's run took about an hour)
      python version7.py --max-iters 2 --batch-size 2000 (quick run)
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


def train(env, policy, value_net, optimizer, value_optimizer, device,
          batch_size=1000, gamma=0.99, seed=2024, max_iters=1000,
          target_reward=200):
    set_seed(seed, env)
    policy.eval()
    eval_rewards = [evaluate(env, policy, device)]
    print(f"Initial Average Reward: {eval_rewards[-1]:.2f}")
    for _ in range(max_iters):
        # collect data: complete trajectories, at least batch_size steps
        start = time.time()
        buffer = []
        while len(buffer) < batch_size:
            trajectory = get_trajectory(env, policy, device, argmax=False)
            buffer.extend(trajectory)
        end = time.time()

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
        value_net.train()
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        acs = torch.tensor(actions, dtype=torch.long).to(device)
        q_vals = torch.tensor(rtgs, dtype=torch.float32).to(device)

        # NEW: advantages with a learned baseline V_phi(s_t).
        # V_phi is trained on normalised rewards-to-go, so compare it with
        # the normalised Q_hat_t (both on the same scale).
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
        scaled_log_likelihoods = selected_log_likelihoods * advantages
        loss = -scaled_log_likelihoods.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        # NEW: update the value network (regression on the targets)
        val_loss = F.mse_loss(vals, targets)
        value_optimizer.zero_grad()
        val_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
        value_optimizer.step()

        policy.eval()
        value_net.eval()
        eval_rewards.append(evaluate(env, policy, device))
        print(f"Iter {len(eval_rewards)-1} - Avg Reward: {eval_rewards[-1]:.2f}"
              f" - Collected {len(buffer)} samples in {end - start:.2f} seconds")

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
        num_layers=args.num_layers,
    ).to(args.device)
    value_net = MLP(
        input_dim=env.observation_space.shape[0],
        hidden_dim=args.hid_dim,
        output_dim=1,
        num_layers=args.num_layers,
    ).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=args.lr)

    eval_rewards = train(env, model, value_net, optimizer, value_optimizer,
                         args.device,
                         batch_size=args.batch_size, gamma=args.gamma,
                         seed=args.seed,
                         max_iters=args.max_iters)
    print("Saved", plot_rewards(eval_rewards, "version7",
                                batch_size=args.batch_size))
    print("Saved", save_model(model, f"policy_{args.env_name}_version7"))

    if args.render:
        env = gym.make(args.env_name, render_mode="human")
        get_trajectory(env, model, args.device, argmax=True, render=True)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--env-name', default='LunarLander-v3')
    parser.add_argument('--hid-dim', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=40_000,
                        help='minimum number of transitions per update')
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--max-iters', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--render', action='store_true',
                        help='watch the trained policy for one episode')
    args = parser.parse_args()
    args.device = get_device()
    main(args)
