import torch
import torch.nn.functional as F
import numpy as np


def vae_loss(recon_x, x, mu, logvar, reconst_loss='mse', a_RECONST=1., a_KLD=1., x_dim=640):
    """Loss function for VAE which consists of reconstruction and KL divergence losses.
    Thanks to https://github.com/pytorch/examples/blob/master/vae/main.py

    You can also balance weights for each loss, just to see what if KLD loss is stronger, etc.

    Args:
        reconst_loss: Reconstruction loss calculation: 'mse' or 'bce'
        a_RECONST: Weight for reconstruction loss.
        a_KLD: Weight for KLD loss.
    """
    func = (F.mse_loss if reconst_loss == 'mse'
            else F.binary_cross_entropy if reconst_loss == 'bce'
            else 'Unknown reconst_loss')
    RECONST = func(recon_x, x.view(-1, x_dim), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return RECONST*a_RECONST + KLD*a_KLD

def lstm_vae_loss(x, x_recon, mu, logvar, step,
                  k=0.0025,
                  x0=2500,
                  anneal_function="logistic"):
    def kl_anneal_function(anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            return float(1/(1 + np.exp(-k*(step-x0))))
        elif anneal_function == 'linear':
            return min(1, step/x0)

    batch_size = x.shape[0]
    # Negative Log Likelihood
    mse_loss = F.mse_loss(x, x_recon, reduction="mean")

    # KL Divergence
    KL_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KL_weight = kl_anneal_function(anneal_function, step, k, x0)

    # return NLL_loss, KL_loss, KL_weight
    total_loss = mse_loss + KL_weight * KL_loss / batch_size
    return total_loss, mse_loss, KL_loss, KL_weight
