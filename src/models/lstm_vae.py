import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import pytorch_lightning as pl

from .losses import lstm_vae_loss


class Encoder(nn.Module):
    def __init__(self, i_dim, h_dim, z_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=i_dim,
                            hidden_size=h_dim,
                            batch_first=True)
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        hn = hn.squeeze()
        mu = self.fc_mu(hn)
        logvar = self.fc_logvar(hn)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, i_dim, z_dim):
        super().__init__()

        self.fc_latent2hidden = nn.Linear(z_dim, i_dim)
        self.lstm = nn.LSTM(input_size=i_dim,
                            hidden_size=i_dim,
                            batch_first=True)

    def forward(self, x, z):
        h0 = self.fc_latent2hidden(z)
        h0 = torch.unsqueeze(h0, 0)
        c0 = torch.zeros_like(h0)
        x_recon, _ = self.lstm(x, (h0, c0))
        return x_recon

class LSTMVAE(pl.LightningModule):
    def __init__(self, i_dim, h_dim, z_dim):
        super().__init__()
        self.encoder = Encoder(i_dim, h_dim, z_dim)
        self.decoder = Decoder(i_dim, z_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        # reparameterize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps

        x_recon = self.decoder(x, z)
        return x_recon, (mu, logvar)

    def training_step(self, batch, batch_idx):
        x, x_shifted = batch
        x_recon, (mu, logvar) = self(x)
        loss = lstm_vae_loss(x_shifted, x_recon, mu, logvar, self.global_step)
        self.log("train/loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, x_shifted = batch
        x_recon, (mu, logvar) = self(x)
        loss = lstm_vae_loss(x_shifted, x_recon, mu, logvar, self.global_step)
        self.log("val/loss", loss)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        return opt
