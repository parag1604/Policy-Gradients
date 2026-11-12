import torch
import torch.nn as nn


class MLP(nn.Module):
    '''
    A plain multi-layer perceptron with tanh activations.

    num_layers counts the Linear layers: num_layers=2 is input -> hidden ->
    output, and every extra layer adds one hidden -> hidden block.
    For a discrete policy the outputs are the action logits.
    '''
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        x = self.fc2(x)
        return x


class RNN(nn.Module):
    '''
    An LSTM over a short history of observations, (batch, hist, obs_dim) or
    (hist, obs_dim), followed by a linear layer on the last hidden state.
    Used in version6.py.
    '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x


class GaussianMLP(MLP):
    '''
    Gaussian policy for continuous actions (version9.py, version10.py):
        pi_theta(a | s) = N(a; mu_theta(s), diag(sigma^2)).
    The MLP outputs the mean mu_theta(s); log sigma is a separate learned
    parameter that does not depend on the state.
    '''
    def __init__(self, input_dim, hidden_dim, action_dim, num_layers=3):
        super(GaussianMLP, self).__init__(
            input_dim, hidden_dim, action_dim, num_layers)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def distribution(self, obs):
        return torch.distributions.Normal(self(obs), self.log_std.exp())
