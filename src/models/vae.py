import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from .losses import vae_loss

class VAE(pl.LightningModule):
    def __init__(self, x_dim, h_dim=400, z_dim=20):
        super().__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.fc11 = nn.Linear(x_dim, h_dim)
        self.fc12 = nn.Linear(h_dim, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
        self.fc31 = nn.Linear(z_dim, h_dim)
        self.fc32 = nn.Linear(h_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, x_dim)

    def encode(self, x):
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc12(x))
        return self.fc21(x), self.fc22(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = F.relu(self.fc31(z))
        h = F.relu(self.fc32(h))
        return torch.sigmoid(self.fc4(h))

    def forward_all(self, x):
        mu, logvar = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), z, mu, logvar

    def forward(self, x):
        yhat, _, _, _ = self.forward_all(x)
        return yhat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, z, mu, logvar = self.forward_all(x)
        loss = vae_loss(recon_x=y_hat, x=x, mu=mu, logvar=logvar,
                                 reconst_loss='mse',
                                 a_RECONST=1.,
                                 a_KLD=.01)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.mse_loss(y_hat, y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
