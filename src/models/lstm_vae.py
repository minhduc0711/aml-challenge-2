import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from .losses import vae_loss

class Encoder(nn.Module):
    def __init__(self, i_dim, h_dim, z_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=i_dim,
                            hidden_size=h_dim,
                            batch_first=True)
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        _, (x, _) = self.lstm(x)
        x = x.squeeze()
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
