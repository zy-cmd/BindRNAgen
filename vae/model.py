import torch
import torch.nn as nn
import torch.nn.functional as F

from vae.encoder import Encoder2D
from vae.decoder import Decoder2D


from vae.encoder import ModifiedEncoder2D
from vae.decoder import ModifiedDecoder2D

class VAE(nn.Module):
    """
    The complete Variational Autoencoder (VAE) model.
    Combines the Encoder and Decoder, and defines the reparameterization trick and loss function.
    """
    def __init__(self, latent_channels=4):
        super(VAE, self).__init__()
        self.encoder = Encoder2D(latent_channels)
        self.decoder = Decoder2D(latent_channels)

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to allow backpropagation through a random node.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var

    def loss_function(self, recon_x, x, mu, log_var, kl_beta=0.01):
        """
        Calculates the VAE loss, which is a sum of reconstruction loss and KL divergence.
        """
        # Reconstruction Loss (scaled by batch size)
        recon_loss = F.smooth_l1_loss(recon_x, x, reduction='sum') / x.shape[0]

        # KL Divergence (scaled by batch size)
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.shape[0]
        
        # Total Loss
        total_loss = recon_loss + kl_beta * kl_div
        
        return recon_loss, kl_div, total_loss




class VAE_V2(nn.Module):
    """
    The complete Variational Autoencoder (VAE) model.
    Combines the Encoder and Decoder, and defines the reparameterization trick and loss function.
    """
    def __init__(self, latent_channels=4):
        super(VAE_V2, self).__init__()
        self.encoder = ModifiedEncoder2D(latent_channels)
        self.decoder = ModifiedDecoder2D(latent_channels)

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to allow backpropagation through a random node.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var

    def loss_function(self, recon_x, x, mu, log_var, kl_beta=0.01):
        """
        Calculates the VAE loss, which is a sum of reconstruction loss and KL divergence.
        """
        # Reconstruction Loss (scaled by batch size)
        recon_loss = F.smooth_l1_loss(recon_x, x, reduction='sum') / x.shape[0]

        # KL Divergence (scaled by batch size)
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.shape[0]
        
        # Total Loss
        total_loss = recon_loss + kl_beta * kl_div
        
        return recon_loss, kl_div, total_loss
