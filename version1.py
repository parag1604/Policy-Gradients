"""
version1 - Maximum-likelihood classification (the starting point).

What this file teaches
----------------------
Before any reinforcement learning, train a small MLP classifier on a
synthetic two-class dataset by maximum likelihood: minimise the negative
log-likelihood -log p_theta(y | x) of the correct labels.

Why start here? The lecture shows that the policy gradient is a
*weighted* maximum-likelihood gradient:

    MLE:  grad = (1/N) sum_i grad log pi_theta(a_i | s_i)
    PG:   grad = (1/N) sum_i grad log pi_theta(a_i | s_i) * (return)

So version2 will reuse exactly this loss (log_softmax + gather the chosen
"label"), with the observed state as x, the action taken as y, and every
term weighted by how good the outcome was.

Lecture: https://parag1604.github.io/notes/Reinforcement-Learning/Policy-Gradients
(section "Policy gradient = weighted maximum likelihood").

Run:  python version1.py            (saves plots/version1.png)
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import MLP
from utils import (train_test_split, get_classification_dataset,
                   plot_decision_boundary, set_seed, get_device)


def train(X_train, y_train, model, optimizer,
          epochs=100, batch_size=32, device=torch.device('cpu')):
    for epoch in range(epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        idxs = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[idxs[i:i + batch_size]]
            y_batch = y_train[idxs[i:i + batch_size]]

            X_batch = torch.tensor(X_batch, dtype=torch.float32).to(device)
            y_batch = torch.tensor(y_batch, dtype=torch.long).to(device)

            outs = model(X_batch)
            # log p_theta(y | x) for every class, then pick the true class.
            # F.nll_loss(log_likelihoods, y) is the same as
            # -log_likelihoods.gather(1, y.unsqueeze(1)).mean()
            log_likelihoods = torch.log_softmax(outs, axis=1)
            loss = F.nll_loss(log_likelihoods, y_batch)
            total_loss += loss.item()
            n_batches += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {total_loss / n_batches:.4f}")


def evaluate(X_test, y_test, model, batch_size, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            X_batch = torch.tensor(X_test[i:i + batch_size],
                                   dtype=torch.float32).to(device)
            y_batch = torch.tensor(y_test[i:i + batch_size],
                                   dtype=torch.long).to(device)
            y_pred = torch.argmax(model(X_batch), axis=1)
            correct += (y_pred == y_batch).sum().item()
    return correct / len(X_test)


def main(args):
    set_seed(args.seed)

    X, y = get_classification_dataset()
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    model = MLP(
        input_dim=X.shape[1],
        hidden_dim=args.hid_dim,
        output_dim=len(np.unique(y))
    ).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(X_train, y_train, model, optimizer,
          args.epochs, args.batch_size, args.device)

    accuracy = evaluate(X_test, y_test, model, args.batch_size, args.device)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Saved", plot_decision_boundary(model, X_test, y_test, 'version1'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hid-dim', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=2025)
    args = parser.parse_args()
    args.device = get_device()
    main(args)
