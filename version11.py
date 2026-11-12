"""
version11 - Off-policy REINFORCE with importance sampling (CartPole-v1).

What this file teaches
----------------------
Every version so far is on-policy: after one gradient step the batch is
thrown away, because it was sampled by the *old* policy. Here we keep a
replay buffer with the last few batches and reuse them. The trajectories in
the buffer were sampled by older (behaviour) policies pi_b, so the gradient
must be corrected with importance weights. With the per-step ratios

    rho_t = pi_theta(a_t | s_t) / pi_b(a_t | s_t)

(the dynamics cancel, so only the policy probabilities are needed, and we
store pi_b(a_t | s_t) when the action is sampled), there are three choices:

  --weights trajectory    the whole-trajectory weight w(tau) = prod_t rho_t
                          on every term. Unbiased, but see below.
  --weights per-decision  causal ("per-decision") weights: the reward r_t'
                          only depends on actions up to t', so it gets the
                          weight rho_{0:t'} = prod_{k<=t'} rho_k:
                            g = sum_t grad log pi(a_t|s_t) *
                                [ sum_{t'>=t} gamma^(t'-t) rho_{0:t'} r_t'
                                  - rho_{0:t} b ]
                          Still unbiased (the baseline term rho_{0:t} b has
                          zero mean), and lower variance than trajectory.
  --weights step          only the current ratio rho_t, i.e. the lecture's
                          "first-order" approximation that ignores the
                          state-distribution mismatch. Biased, but its
                          variance does not grow with the horizon; it is
                          fine as long as pi_theta stays close to pi_b
                          (the idea behind TRPO/PPO). This is what the
                          original course code did.

Why the variance grows with the horizon: log w(tau) = sum_t log rho_t is a
sum of T random terms. E_b[rho_t] = 1 for every step, so E[w] = 1, but
E_b[rho_t^2] >= 1 (Jensen), and for roughly independent steps
Var[w] = prod_t E_b[rho_t^2] - 1, which grows exponentially in T. In
practice a few trajectories get enormous weights and the rest get almost
none: the effective sample size (ESS = (sum w)^2 / sum w^2, printed every
update) collapses. On CartPole-v1 episodes last up to 500 steps, so this
is easy to see.

--clip c caps every (cumulative) weight at c ("truncated importance
sampling"). This bounds the variance at the price of some bias. Beware: the
baseline term -rho_{0:t} b has zero mean only with the exact weights. Capped
weights shrink it for the actions the current policy now prefers, so it
keeps pushing their probability up; on CartPole with 5 gradient steps per
batch this collapsed 3 of 5 seeds to a deterministic policy (see README).

Bugs fixed relative to the original course code
-----------------------------------------------
  - The buffer was cut to its last buffer_size transitions, which could
    chop the beginning off the oldest episode; now whole episodes are
    dropped, so every trajectory's weights start at t = 0.
  - Only the single-step ratio was available; the trajectory and
    per-decision weights are added.
  - The ratio was exp(log pi) / pi_b in float64 and float32 mixed; it is now
    computed as exp(log pi - log pi_b) in log space, which cannot overflow
    before clipping.
  - Ported from the old gym API (env.seed, 4-tuple step) to gymnasium, and
    the script no longer calls sys.exit() when the task is solved.

With --buffer-batches 1 every weight is exactly 1 and the update is the
on-policy REINFORCE of version5 (reward-to-go, baseline, normalisation);
--compare runs that and the three weightings over several seeds and plots
the learning curves together.

Based on Parag's original off-policy REINFORCE from the E0 270 course code.

Lecture: https://parag1604.github.io/notes/Reinforcement-Learning/Policy-Gradients
(section "Off-policy Policy Gradients").

Run:  python version11.py [--weights per-decision] [--clip 10] [--max-iters 5]
      python version11.py --compare      (all variants, 3 seeds, ~10 min on a CPU)
"""
import argparse
import os
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from model import MLP
from utils import set_seed, plot_rewards, save_model, get_device


def get_trajectory(env, policy, device, argmax=True, render=False):
    '''
    Run one episode. argmax=True acts greedily (for evaluation);
    argmax=False samples a ~ pi(. | s) (for training).
    NEW: also returns log pi_b(a_t | s_t) of every sampled action, the
    behaviour probability that the importance weights divide by.
    '''
    obs, _ = env.reset()
    done = False
    observations, actions, rewards, logps = [], [], [], []
    while not done:
        if render:
            time.sleep(0.005)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = policy(obs_tensor)
            if argmax:
                action = torch.argmax(logits).item()
            else:
                action = torch.distributions.Categorical(
                    logits=logits).sample().item()
            logp = torch.log_softmax(logits, dim=-1)[action].item()
        n_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        logps.append(logp)
        obs = n_obs
    return {'obs': np.array(observations, dtype=np.float32),
            'actions': np.array(actions),
            'rewards': np.array(rewards, dtype=np.float64),
            'behaviour_logp': np.array(logps, dtype=np.float64)}


def evaluate(env, policy, device, num_runs=10):
    '''Average undiscounted return of the greedy policy.'''
    with torch.no_grad():
        policy.eval()
        return np.mean([get_trajectory(env, policy, device)['rewards'].sum()
                        for _ in range(num_runs)])


def discounted_cumsum(x, gamma):
    '''y_t = sum_{t' >= t} gamma^(t'-t) x_t', computed backwards.'''
    y = np.zeros_like(x)
    running = 0.0
    for t in range(len(x) - 1, -1, -1):
        running = x[t] + gamma * running
        y[t] = running
    return y


def episode_terms(episode, log_pi, gamma, weights, clip):
    '''
    For one episode, return the per-step coefficients that multiply
    grad log pi(a_t | s_t), split as (weighted return, weight on baseline),
    so that coef_t = ret_t - base_w_t * b.

    log_pi: current log pi_theta(a_t | s_t) (numpy, no gradient).
    '''
    rewards = episode['rewards']
    log_rho = log_pi - episode['behaviour_logp']          # log rho_t
    log_clip = np.inf if clip is None else np.log(clip)
    if weights == 'trajectory':
        # one weight for the whole trajectory, w(tau) = prod_t rho_t
        w = np.exp(min(log_rho.sum(), log_clip)) * np.ones_like(rewards)
        return w * discounted_cumsum(rewards, gamma), w
    if weights == 'per-decision':
        # rho_{0:t} = prod_{k <= t} rho_k; reward r_t' carries rho_{0:t'}
        w = np.exp(np.minimum(np.cumsum(log_rho), log_clip))
        return discounted_cumsum(w * rewards, gamma), w
    if weights == 'step':
        # first-order approximation: only the current ratio rho_t
        w = np.exp(np.minimum(log_rho, log_clip))
        return w * discounted_cumsum(rewards, gamma), w
    raise ValueError(weights)


def effective_sample_size(w):
    '''Kish's ESS, (sum w)^2 / sum w^2: N for equal weights, 1 if one dominates.'''
    return w.sum() ** 2 / max((w ** 2).sum(), 1e-300)


def train(env, policy, optimizer, device, batch_size=1000, gamma=0.99,
          weights='per-decision', clip=None, buffer_batches=3,
          updates_per_batch=1, seed=2024, max_iters=1000, target_reward=480,
          verbose=True):
    set_seed(seed, env)
    policy.eval()
    eval_rewards = [evaluate(env, policy, device)]
    ess_fractions = []
    log_weights = []   # (episode length T, log w(tau)) pairs, for plotting
    if verbose:
        print(f"Initial Average Reward: {eval_rewards[-1]:.2f}")
    batches = []       # NEW: replay buffer, a list of batches of whole episodes
    for _ in range(max_iters):
        # collect data with the current policy (it becomes a behaviour policy)
        batch, n = [], 0
        while n < batch_size:
            episode = get_trajectory(env, policy, device, argmax=False)
            batch.append(episode)
            n += len(episode['rewards'])
        batches = (batches + [batch])[-buffer_batches:]
        episodes = [ep for b in batches for ep in b]

        X = torch.tensor(np.concatenate([ep['obs'] for ep in episodes])).to(device)
        A = torch.tensor(np.concatenate([ep['actions'] for ep in episodes]),
                         dtype=torch.long).to(device)
        # constant baseline: the average (unweighted) reward-to-go in the buffer
        b = np.concatenate([discounted_cumsum(ep['rewards'], gamma)
                            for ep in episodes]).mean()

        policy.train()
        for _ in range(updates_per_batch):
            log_pi = torch.log_softmax(policy(X), dim=1).gather(
                1, A.view(-1, 1)).squeeze(1)
            # importance weights use the current policy, but no gradient
            # flows through them: grad (pi/pi_b) = (pi/pi_b) grad log pi
            log_pi_np = log_pi.detach().cpu().numpy().astype(np.float64)
            coefs, log_w_traj, start = [], [], 0
            for ep in episodes:
                T = len(ep['rewards'])
                ret, base_w = episode_terms(ep, log_pi_np[start:start + T],
                                            gamma, weights, clip)
                coefs.append(ret - base_w * b)
                log_w_traj.append(log_pi_np[start:start + T].sum()
                                  - ep['behaviour_logp'].sum())
                start += T
            coefs = np.concatenate(coefs)
            # normalise the scale only (as in version5); the sign is kept
            coefs = coefs / (coefs.std() + 1e-8)
            coefs = torch.tensor(coefs, dtype=torch.float32).to(device)
            loss = -(log_pi * coefs).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

        # diagnostic, whatever --weights is: ESS of the unclipped trajectory
        # weights w(tau) = prod_t rho_t (after the last gradient step), as a
        # fraction of the number of episodes in the buffer
        log_w = np.array(log_w_traj)
        w_traj = np.exp(log_w - log_w.max())       # ESS is scale-invariant
        ess_fractions.append(effective_sample_size(w_traj) / len(episodes))
        log_weights.extend((len(ep['rewards']), lw)
                           for ep, lw in zip(episodes, log_w))

        policy.eval()
        eval_rewards.append(evaluate(env, policy, device))
        if verbose:
            print(f"Iter {len(eval_rewards)-1} - Avg Reward: "
                  f"{eval_rewards[-1]:.2f} - episodes in buffer: "
                  f"{len(episodes)} - ESS: {100 * ess_fractions[-1]:.0f}%")

        if len(eval_rewards) > 10 and np.mean(eval_rewards[-10:]) > target_reward:
            break
    return eval_rewards, ess_fractions, log_weights


def make_and_train(args, seed, weights, buffer_batches, clip, verbose,
                   updates_per_batch=None, target_reward=480):
    env = gym.make(args.env_name)
    set_seed(seed)
    model = MLP(
        input_dim=env.observation_space.shape[0],
        hidden_dim=args.hid_dim,
        output_dim=env.action_space.n,
    ).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if updates_per_batch is None:
        updates_per_batch = args.updates_per_batch
    eval_rewards, ess, log_weights = train(
        env, model, optimizer, args.device, batch_size=args.batch_size,
        gamma=args.gamma, weights=weights, clip=clip,
        buffer_batches=buffer_batches,
        updates_per_batch=updates_per_batch, seed=seed,
        max_iters=args.max_iters, target_reward=target_reward,
        verbose=verbose)
    return env, model, eval_rewards, ess, log_weights


def compare(args, save_path='plots'):
    '''
    Learning curves of on-policy REINFORCE (version5's update: buffer of one
    batch, one gradient step) against the off-policy variants, which keep
    --buffer-batches batches and take --updates-per-batch steps on them.
    Every run is trained for exactly --max-iters updates (no early stop).
    '''
    K, nb = args.updates_per_batch, args.buffer_batches
    variants = [
        ('on-policy REINFORCE (1 batch, 1 step)', 'step', 1, None, 1),
        ('off-policy, step ratio (first-order)', 'step', nb, None, K),
        ('off-policy, trajectory weights', 'trajectory', nb, None, K),
        ('off-policy, per-decision weights', 'per-decision', nb, None, K),
        (f'off-policy, per-decision, clipped at {args.compare_clip:g}',
         'per-decision', nb, args.compare_clip, K),
    ]
    seeds = [args.seed + i for i in range(args.num_seeds)]
    fig, axes = plt.subplots(3, 1, figsize=(8, 11),
                             gridspec_kw={'height_ratios': [2, 1, 1.2]})
    ax, ax2, ax3 = axes
    for label, weights, nbuf, clip, k in variants:
        curves, esses = [], []
        for seed in seeds:
            t0 = time.time()
            _, _, rewards, ess, log_w = make_and_train(
                args, seed, weights, nbuf, clip, verbose=False,
                updates_per_batch=k, target_reward=np.inf)
            curves.append(rewards)
            esses.append(ess)
            first = next((i for i, r in enumerate(rewards) if r >= 475), None)
            print(f"{label:48s} seed {seed}: first >= 475 at update {first}, "
                  f"mean of last 10 = {np.mean(rewards[-10:]):.0f}, "
                  f"{time.time() - t0:.0f} s", flush=True)
            if weights == 'per-decision' and clip is None and seed == seeds[0]:
                scatter = np.array(log_w)
        curves = np.array(curves)
        x = np.arange(curves.shape[1])
        mean, std = curves.mean(0), curves.std(0)
        line, = ax.plot(x, mean, label=label)
        ax.fill_between(x, np.clip(mean - std, 0, None), np.clip(mean + std, None, 500),
                        alpha=0.12, color=line.get_color())
        ax2.plot(np.arange(1, curves.shape[1]), 100 * np.mean(esses, 0),
                 color=line.get_color())
        print(f"{label:48s} mean of last 10 updates over seeds: "
              f"{curves[:, -10:].mean():.0f}", flush=True)
    ax.set_ylabel('greedy-policy return\n(mean +- 1 std over seeds)')
    ax.set_title(f'version11: off-policy REINFORCE on {args.env_name}\n'
                 f'{len(seeds)} seeds; off-policy runs reuse the last {nb} batches '
                 f'with {K} gradient steps per batch', fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)
    for a in (ax, ax2):
        a.set_xlabel(f'update (x{args.batch_size} new transitions)')
    ax2.set_ylabel('ESS of w(tau)\n(% of buffer episodes)')
    ax2.set_yscale('log')
    ax2.grid(alpha=0.3)
    # why the variance grows with the horizon: log w(tau) is a sum of T terms
    ax3.scatter(scatter[:, 0], scatter[:, 1] / np.log(10), s=3, alpha=0.3)
    ax3.axhline(0, color='k', linewidth=0.8)
    ax3.set_xlabel('episode length T')
    ax3.set_ylabel('log10 w(tau)')
    ax3.set_title('trajectory weights in the buffer (per-decision run, first seed): '
                  'the spread grows with T', fontsize=9)
    ax3.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, f'{args.plot_name}.png')
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def main(args):
    if args.compare:
        print("Saved", compare(args))
        return
    env, model, eval_rewards, _, _ = make_and_train(
        args, args.seed, args.weights, args.buffer_batches, args.clip,
        verbose=True)
    print("Saved", plot_rewards(eval_rewards, "version11",
                                batch_size=args.batch_size))
    print("Saved", save_model(model, f"policy_{args.env_name}_version11"))

    if args.render:
        env = gym.make(args.env_name, render_mode="human")
        get_trajectory(env, model, args.device, argmax=True, render=True)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--env-name', default='CartPole-v1')
    parser.add_argument('--hid-dim', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='minimum number of NEW transitions per update')
    parser.add_argument('--buffer-batches', type=int, default=3,
                        help='how many of the latest batches to train on '
                             '(1 = on-policy)')
    parser.add_argument('--updates-per-batch', type=int, default=1,
                        help='gradient steps on the buffer per new batch')
    parser.add_argument('--weights', default='per-decision',
                        choices=['trajectory', 'per-decision', 'step'])
    parser.add_argument('--clip', type=float, default=None,
                        help='cap every importance weight at this value')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--max-iters', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--compare', action='store_true',
                        help='run on-policy and all weightings, plot together')
    parser.add_argument('--num-seeds', type=int, default=5)
    parser.add_argument('--compare-clip', type=float, default=10.0)
    parser.add_argument('--plot-name', default='version11',
                        help='file name (without .png) of the --compare plot')
    parser.add_argument('--render', action='store_true',
                        help='watch the trained policy for one episode')
    args = parser.parse_args()
    args.device = get_device()
    main(args)
