import os
import random
from typing import Tuple, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch


def get_device():
    '''
    cuda if available, otherwise cpu.

    Apple's mps backend is skipped on purpose: these networks are tiny and
    are called once per environment step, so the per-call overhead of mps
    dominates (an actor-critic epoch took about 23 s on mps vs 0.4 s on cpu).
    '''
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def set_seed(seed: int, env: Optional[gym.Env] = None) -> None:
    '''
    Set the seed for the random number generators.

    Args:
        seed: the seed for the random number generator
        env: if given, only the environment (or vector environment) is seeded
    '''
    if env is not None:
        # gymnasium seeds an environment through its first reset
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_classification_dataset(
        var: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Generate a synthetic dataset with two Gaussian classes, 1 and 0.

    Args:
        var: used in the covariance of the gaussians

    Returns:
        X: the dataset
        y: the labels
    '''
    mean1 = [-2, 2]
    mean2 = [1, -1]
    cov = [[var, 0], [0, var]]
    cluster1 = np.random.multivariate_normal(mean1, cov, 100)
    cluster2 = np.random.multivariate_normal(mean2, cov, 100)
    X = np.vstack((cluster1, cluster2))
    y = np.concatenate((np.ones(100, dtype=int), np.zeros(100, dtype=int)))
    return X, y


def train_test_split(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        seed=2025
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''Shuffle and split the dataset into train and test sets.'''
    assert 0.0 < test_size < 1.0
    indices = np.arange(X.shape[0])
    np.random.RandomState(seed).shuffle(indices)
    n_test = int(test_size * X.shape[0])
    train_idx, test_idx = indices[:-n_test], indices[-n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def plot_decision_boundary(
        model,
        X: np.ndarray,
        y: np.ndarray,
        title: str = 'version1',
        save_path: str = 'plots',
) -> str:
    '''Plot the decision boundary of a classifier and save it to save_path.'''
    os.makedirs(save_path, exist_ok=True)
    device = next(model.parameters()).device
    fig, ax = plt.subplots()
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    with torch.no_grad():
        Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()],
                               dtype=torch.float32).to(device)).argmax(axis=1)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z.cpu().numpy(), alpha=0.3, cmap='Paired_r')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k',
                         marker='o', cmap='Paired_r', s=50)
    ax.add_artist(ax.legend(*scatter.legend_elements(), title="Classes"))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    path = os.path.join(save_path, f'{title}.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_rewards(
        rewards,
        title: str = 'Rewards',
        xlabel: str = 'Iteration',
        ylabel: str = 'Reward',
        save_path: str = 'plots',
        batch_size: int = 1000
) -> str:
    '''
    Plot the evaluation reward after every policy update, with a 5-point
    moving average, and save it as save_path/<title>.png.
    '''
    os.makedirs(save_path, exist_ok=True)
    rewards = np.asarray(rewards, dtype=np.float32)
    plt.plot(rewards, label='Average Reward', alpha=0.5)
    if len(rewards) >= 5:
        rewards_ma = np.convolve(rewards, np.ones(5) / 5, mode='valid')
        plt.plot(np.arange(4, len(rewards_ma) + 4), rewards_ma,
                 label='Moving Average')
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel + f" (x{batch_size} transitions)")
    plt.ylabel(ylabel)
    plt.grid()
    path = os.path.join(save_path, f'{title}.png')
    plt.savefig(path)
    plt.close()
    return path


def save_model(model, name: str, save_dir: str = 'models') -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{name}.pth')
    torch.save(model.state_dict(), path)
    return path
