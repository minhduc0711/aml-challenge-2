import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, flatten_dim, z_dim):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, 5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 64, 5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 64, 5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Flatten(),
            nn.Linear(flatten_dim, z_dim)
        )
        
    def forward(self, x):
        return self.stack(x)
    

class Decoder(nn.Module):
    def __init__(self, z_dim, shape_unflatten):
        super().__init__()
        self.shape_unflatten = shape_unflatten
        self.fc = nn.Linear(z_dim, np.prod(shape_unflatten))
        self.stack = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 64, 5, stride=2, output_padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
                
            nn.ConvTranspose2d(64, 64, 5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
                       
            nn.ConvTranspose2d(64, 32, 5, stride=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.ConvTranspose2d(32, 1, 5, stride=2, output_padding=1),
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, *self.shape_unflatten)
        x = self.stack(x)
        return x
    
    
class CNNAE(pl.LightningModule):
    def __init__(self, shape_preflatten=(64, 14, 3), 
                  z_dim=1024):
        super().__init__()
        self.encoder = Encoder(np.prod(shape_preflatten), z_dim)
        self.decoder = Decoder(z_dim, shape_preflatten)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def training_step(self, batch, batch_idx):
        x = batch["input"]
        x_recon = self(x)
        
        loss = self.loss_fn(x, x_recon)
        self.log("train/loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch["input"]
        x_recon = self(x)
        
        loss = self.loss_fn(x, x_recon)
        self.log("val/loss", loss)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        return opt
