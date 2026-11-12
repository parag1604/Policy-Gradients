"""
version12 - Proximal Policy Optimization (PPO): reuse each batch for K
epochs with a clipped surrogate objective (CartPole-v1, InvertedPendulum-v5).

What this file teaches
----------------------
version11 reused old data with importance weights and showed the two ways
this goes wrong: products of ratios have a variance that explodes with the
horizon, and capping the weights biases the gradient (with a baseline it
collapsed several seeds). PPO (Schulman et al., 2017) keeps only the
per-step ratio against the policy that collected the batch,

    rho_t(theta) = pi_theta(a_t | s_t) / pi_old(a_t | s_t),

i.e. version11's "first-order" approximation, and makes it safe by never
letting the new policy move far from pi_old. It maximises the clipped
surrogate

    L(theta) = E_t[ min( rho_t A_t, clip(rho_t, 1 - eps, 1 + eps) A_t ) ]

(eps = 0.2). The gradient of the unclipped term is rho_t A_t grad log pi,
the importance-weighted policy gradient. The min with the clipped term
switches that gradient OFF for a sample once its ratio has left
[1 - eps, 1 + eps] in the direction that would improve the objective:
for A_t > 0 the objective stops rewarding rho_t above 1 + eps, for A_t < 0
it stops rewarding rho_t below 1 - eps. Moves in the other direction (which
make the objective worse) are never clipped, so L is a pessimistic lower
bound of the unclipped surrogate.

Clipping the OBJECTIVE is not clipping the WEIGHTS (version11 --clip). A
clipped weight still multiplies grad log pi, so every sample keeps pushing,
just with a biased size, and nothing stops the policy from drifting further
and further from the data. A clipped objective has zero gradient for the
samples that already moved enough, so the policy stops moving on them:
large ratios stop contributing instead of contributing a wrong amount.

Other ingredients (all standard):
  - A learned value function V_phi (as in version7) and GAE(lambda)
    advantages, A_t = sum_l (gamma lambda)^l delta_{t+l} with
    delta_t = r_t + gamma V(s_{t+1}) - V(s_t). At a time-limit truncation we
    bootstrap with V of the last observation; at a real termination V = 0.
    The advantages are normalised per batch.
  - K epochs of minibatch Adam steps on the same batch (policy loss +
    value loss, optional entropy bonus).
  - Two diagnostics, printed every update: the clip fraction (share of
    samples whose ratio left [1 - eps, 1 + eps] in the last epoch) and the
    approximate KL(pi_old || pi_new) = E[(rho - 1) - log rho].

--compare trains, with the same seeds and the same new transitions per
update, on-policy REINFORCE with a learned baseline (version7's update),
version11's per-decision importance sampling with K full-batch steps on the
batch, and PPO with K = 5 and K = 10, and plots return vs environment steps.

Lecture: https://parag1604.github.io/notes/Reinforcement-Learning/Policy-Gradients
(section "Off-policy Policy Gradients").

Run:  python version12.py                       (PPO on CartPole-v1)
      python version12.py --env-name InvertedPendulum-v5
      python version12.py --compare             (all methods, 5 seeds)
"""
import argparse
import os
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import MLP, GaussianMLP
from utils import set_seed, plot_rewards, save_model, get_device
from version11 import discounted_cumsum, episode_terms

# per-environment defaults: transitions per update, discount, the learning
# rate of the single-step methods (versions 7 and 9/11), training length
ENV_DEFAULTS = {
    'CartPole-v1': dict(batch_size=1000, gamma=0.99, pg_lr=0.01, max_iters=60),
    'InvertedPendulum-v5': dict(batch_size=2000, gamma=0.99, pg_lr=0.03,
                                max_iters=40),
}


class Policy:
    '''A thin wrapper so the same code handles discrete and Gaussian policies.'''
    def __init__(self, env, hid_dim, device):
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        obs_dim = env.observation_space.shape[0]
        if self.discrete:
            self.net = MLP(obs_dim, hid_dim, env.action_space.n).to(device)
        else:
            self.net = GaussianMLP(obs_dim, hid_dim,
                                   env.action_space.shape[0]).to(device)
            self.a_max = env.action_space.high
        self.device = device

    def dist(self, obs):
        if self.discrete:
            return torch.distributions.Categorical(logits=self.net(obs))
        return self.net.distribution(obs)

    def log_prob(self, obs, actions):
        lp = self.dist(obs).log_prob(actions)
        return lp if self.discrete else lp.sum(-1)

    def entropy(self, obs):
        ent = self.dist(obs).entropy()
        return ent if self.discrete else ent.sum(-1)

    def act(self, obs, greedy):
        '''Returns (action stored for learning, action sent to env, log pi).'''
        obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            d = self.dist(obs_t)
            if self.discrete:
                a = d.probs.argmax() if greedy else d.sample()
                return a.item(), a.item(), d.log_prob(a).item()
            a = d.mean if greedy else d.sample()
            lp = d.log_prob(a).sum().item()
            a = a.cpu().numpy()
            # squash into the bounds (version9); log pi is of the raw sample
            return a, np.tanh(a) * self.a_max, lp


def run_episode(env, policy, greedy):
    obs, _ = env.reset()
    ep = {'obs': [], 'actions': [], 'rewards': [], 'behaviour_logp': []}
    done = False
    while not done:
        a, a_env, lp = policy.act(obs, greedy)
        n_obs, r, terminated, truncated, _ = env.step(a_env)
        ep['obs'].append(obs)
        ep['actions'].append(a)
        ep['rewards'].append(r)
        ep['behaviour_logp'].append(lp)
        obs = n_obs
        done = terminated or truncated
    ep = {k: np.array(v) for k, v in ep.items()}
    ep['obs'] = ep['obs'].astype(np.float32)
    ep['rewards'] = ep['rewards'].astype(np.float64)
    ep['terminated'] = terminated
    ep['last_obs'] = obs.astype(np.float32)
    return ep


def evaluate(env, policy, num_runs=5):
    '''Average undiscounted return of the greedy (mean) policy.'''
    return np.mean([run_episode(env, policy, greedy=True)['rewards'].sum()
                    for _ in range(num_runs)])


def gae(ep, value_net, gamma, lam, device):
    '''GAE(lambda) advantages and value targets (A + V) for one episode.'''
    with torch.no_grad():
        v = value_net(torch.tensor(ep['obs']).to(device)).squeeze(-1).cpu().numpy()
        v_last = 0.0 if ep['terminated'] else value_net(
            torch.tensor(ep['last_obs']).to(device)).item()
    v_next = np.append(v[1:], v_last)
    deltas = ep['rewards'] + gamma * v_next - v
    adv = discounted_cumsum(deltas, gamma * lam)
    return adv, adv + v


def kl_and_clipfrac(log_ratio, eps):
    ratio = log_ratio.exp()
    approx_kl = ((ratio - 1) - log_ratio).mean().item()
    clipfrac = ((ratio - 1).abs() > eps).float().mean().item()
    return approx_kl, clipfrac


def train(env, policy, value_net, method, args, seed, verbose=True):
    '''
    method: 'reinforce' (version7: one step on rewards-to-go minus V),
            'is'        (version11 per-decision IS, K full-batch steps),
            'ppo'       (clipped surrogate, K epochs of minibatches).
    Returns the evaluation returns and per-update diagnostics.
    '''
    device = policy.device
    set_seed(seed, env)
    lr = args.lr if method == 'ppo' else args.pg_lr
    pi_opt = optim.Adam(policy.net.parameters(), lr=lr)
    v_opt = optim.Adam(value_net.parameters(), lr=args.value_lr)
    hist = {'eval': [evaluate(env, policy)], 'steps': [0], 'kl': [],
            'clipfrac': []}
    steps = 0
    for it in range(args.max_iters):
        episodes, n = [], 0
        while n < args.batch_size:
            episodes.append(run_episode(env, policy, greedy=False))
            n += len(episodes[-1]['rewards'])
        steps += n

        obs = torch.tensor(np.concatenate([e['obs'] for e in episodes])).to(device)
        acts = np.concatenate([e['actions'] for e in episodes])
        acts = torch.tensor(acts, dtype=torch.long if policy.discrete
                            else torch.float32).to(device)
        old_logp = torch.tensor(np.concatenate(
            [e['behaviour_logp'] for e in episodes]), dtype=torch.float32).to(device)
        rtgs = np.concatenate([discounted_cumsum(e['rewards'], args.gamma)
                               for e in episodes])

        if method == 'reinforce':
            # version7: V_phi regressed on normalised rewards-to-go
            targets = torch.tensor((rtgs - rtgs.mean()) / (rtgs.std() + 1e-8),
                                   dtype=torch.float32).to(device)
            vals = value_net(obs).squeeze(-1)
            adv = targets - vals.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            loss = -(policy.log_prob(obs, acts) * adv).mean()
            pi_opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.net.parameters(), 1.0)
            pi_opt.step()
            v_loss = F.mse_loss(vals, targets)
            v_opt.zero_grad(); v_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
            v_opt.step()

        elif method == 'is':
            # version11: per-decision weights against the collecting policy,
            # constant baseline, K full-batch steps on the same batch
            b = rtgs.mean()
            for _ in range(args.epochs):
                logp = policy.log_prob(obs, acts)
                logp_np = logp.detach().cpu().numpy().astype(np.float64)
                coefs, start = [], 0
                for e in episodes:
                    T = len(e['rewards'])
                    ret, w = episode_terms(e, logp_np[start:start + T],
                                           args.gamma, 'per-decision', None)
                    coefs.append(ret - w * b)
                    start += T
                coefs = np.concatenate(coefs)
                coefs = torch.tensor(coefs / (coefs.std() + 1e-8),
                                     dtype=torch.float32).to(device)
                loss = -(logp * coefs).mean()
                pi_opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.net.parameters(), 1.0)
                pi_opt.step()

        elif method == 'ppo':
            advs, rets = zip(*[gae(e, value_net, args.gamma, args.lam, device)
                               for e in episodes])
            adv = torch.tensor(np.concatenate(advs), dtype=torch.float32).to(device)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ret = torch.tensor(np.concatenate(rets), dtype=torch.float32).to(device)
            N = len(adv)
            for _ in range(args.epochs):
                perm = torch.randperm(N, device=device)
                for i in range(0, N, args.minibatch_size):
                    idx = perm[i:i + args.minibatch_size]
                    logp = policy.log_prob(obs[idx], acts[idx])
                    ratio = (logp - old_logp[idx]).exp()
                    surr = torch.min(
                        ratio * adv[idx],
                        ratio.clamp(1 - args.eps, 1 + args.eps) * adv[idx])
                    loss = -surr.mean()
                    if args.ent_coef > 0:
                        loss = loss - args.ent_coef * policy.entropy(obs[idx]).mean()
                    v_loss = F.mse_loss(value_net(obs[idx]).squeeze(-1), ret[idx])
                    pi_opt.zero_grad(); v_opt.zero_grad()
                    (loss + v_loss).backward()
                    torch.nn.utils.clip_grad_norm_(policy.net.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
                    pi_opt.step(); v_opt.step()
        else:
            raise ValueError(method)

        # diagnostics on the whole batch, after this update, against pi_old
        with torch.no_grad():
            kl, cf = kl_and_clipfrac(policy.log_prob(obs, acts) - old_logp,
                                     args.eps)
        hist['kl'].append(kl)
        hist['clipfrac'].append(cf)
        hist['eval'].append(evaluate(env, policy))
        hist['steps'].append(steps)
        if verbose:
            print(f"Iter {it + 1} - steps {steps} - Avg Reward: "
                  f"{hist['eval'][-1]:.1f} - approx KL {kl:.4f} - "
                  f"clip fraction {cf:.2f}", flush=True)
    return hist


def make_and_train(args, method, epochs, seed, verbose):
    env = gym.make(args.env_name)
    set_seed(seed)
    policy = Policy(env, args.hid_dim, args.device)
    value_net = MLP(env.observation_space.shape[0], args.hid_dim, 1).to(args.device)
    a = argparse.Namespace(**vars(args))
    a.epochs = epochs
    return env, policy, train(env, policy, value_net, method, a, seed, verbose)


VARIANTS = {
    'reinforce': ('REINFORCE + learned baseline (1 step)', 'reinforce', 1),
    'is5': ('per-decision IS, K = 5 steps (version11)', 'is', 5),
    'ppo5': ('PPO, K = 5 epochs', 'ppo', 5),
    'ppo10': ('PPO, K = 10 epochs', 'ppo', 10),
}


def run_variants(args, names):
    seeds = [args.seed + i for i in range(args.num_seeds)]
    os.makedirs('plots', exist_ok=True)
    for name in names:
        label, method, K = VARIANTS[name]
        runs = []
        for seed in seeds:
            t0 = time.time()
            _, _, h = make_and_train(args, method, K, seed, verbose=False)
            runs.append(h)
            print(f"{args.env_name} {label:42s} seed {seed}: final eval "
                  f"{h['eval'][-1]:.0f}, mean of last 10 "
                  f"{np.mean(h['eval'][-10:]):.0f}, median KL "
                  f"{np.median(h['kl']):.4f}, median clip frac "
                  f"{np.median(h['clipfrac']):.2f}, {time.time() - t0:.0f} s",
                  flush=True)
        np.savez(f'plots/version12_{args.env_name}_{name}.npz',
                 **{k: np.array([r[k] for r in runs]) for k in runs[0]})


def plot_comparison(args, names):
    fig, axes = plt.subplots(3, 1, figsize=(8, 10),
                             gridspec_kw={'height_ratios': [2, 1, 1]})
    for name in names:
        path = f'plots/version12_{args.env_name}_{name}.npz'
        if not os.path.exists(path):
            continue
        d = np.load(path)
        label = VARIANTS[name][0]
        steps = d['steps'].mean(0)                 # (seeds, iters + 1)
        mean, std = d['eval'].mean(0), d['eval'].std(0)
        line, = axes[0].plot(steps, mean, label=label)
        # the band is clipped to the possible returns [0, best seen]
        axes[0].fill_between(steps, np.clip(mean - std, 0, None),
                             np.minimum(mean + std, d['eval'].max()),
                             alpha=0.12, color=line.get_color())
        axes[1].plot(steps[1:], np.median(d['kl'], 0), color=line.get_color())
        axes[2].plot(steps[1:], np.median(d['clipfrac'], 0),
                     color=line.get_color())
        n_seeds = d['eval'].shape[0]
    axes[0].set_title(f'version12 on {args.env_name}: {n_seeds} seeds, '
                      f'{args.batch_size} new transitions per update',
                      fontsize=10)
    axes[0].set_ylabel('greedy-policy return\n(mean +- 1 std over seeds)')
    axes[0].legend(fontsize=8, loc='lower right')
    axes[1].set_ylabel('approx. KL(pi_old||pi_new)\nper update (median)')
    axes[1].set_yscale('log')
    axes[2].set_ylabel(f'fraction of ratios\noutside [1-eps, 1+eps]')
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xlabel('environment steps')
    fig.tight_layout()
    path = f'plots/version12_{args.env_name}.png'
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def main(args):
    if args.compare or args.plot_only:
        names = args.variants.split(',')
        if not args.plot_only:
            run_variants(args, names)
        print("Saved", plot_comparison(args, list(VARIANTS)))
        return
    env, policy, h = make_and_train(args, 'ppo', args.epochs, args.seed,
                                    verbose=True)
    print("Saved", plot_rewards(h['eval'], "version12",
                                batch_size=args.batch_size))
    print("Saved", save_model(policy.net, f"policy_{args.env_name}_version12"))
    if args.render:
        env = gym.make(args.env_name, render_mode="human")
        run_episode(env, policy, greedy=True)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--env-name', default='CartPole-v1',
                        choices=list(ENV_DEFAULTS))
    parser.add_argument('--hid-dim', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=None,
                        help='new transitions per update (default per env)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='K: PPO epochs over each batch')
    parser.add_argument('--minibatch-size', type=int, default=64)
    parser.add_argument('--eps', type=float, default=0.2,
                        help='PPO clip range')
    parser.add_argument('--lam', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--lr', type=float, default=3e-4, help='PPO policy lr')
    parser.add_argument('--pg-lr', type=float, default=None,
                        help='lr of the REINFORCE and IS baselines')
    parser.add_argument('--value-lr', type=float, default=1e-3)
    parser.add_argument('--ent-coef', type=float, default=0.0,
                        help='entropy bonus (PPO only)')
    parser.add_argument('--max-iters', type=int, default=None)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--plot-only', action='store_true',
                        help='re-plot saved --compare results')
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument('--num-seeds', type=int, default=5)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    for k, v in ENV_DEFAULTS[args.env_name].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    args.device = get_device()
    main(args)
