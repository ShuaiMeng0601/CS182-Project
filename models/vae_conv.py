# models/vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    def __init__(self, seq_len=60, action_dim=10, channels=[64,128,256], latent_dim=64):
        super().__init__()

        layers = []
        in_ch = action_dim
        length = seq_len

        for ch in channels:
            layers.append(nn.Conv1d(in_ch, ch, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU())
            in_ch = ch
            length = (length + 1) // 2
            
        self.conv = nn.Sequential(*layers)
        self.final_len = length

        self.fc_mu = nn.Linear(channels[-1] * length, latent_dim)
        self.fc_logvar = nn.Linear(channels[-1] * length, latent_dim)

    def forward(self, x):       # x: (B, seq_len, 10)
        x = x.permute(0,2,1)    # → (B,10,seq_len)
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)



class ConvDecoder(nn.Module):
    def __init__(self, seq_len=60, action_dim=10, channels=[256,128,64], latent_dim=64):
        super().__init__()

        self.init_len = seq_len // (2 ** len(channels))
        self.fc = nn.Linear(latent_dim, channels[0] * self.init_len)

        layers = []
        for i in range(len(channels)-1):
            layers.append(nn.ConvTranspose1d(
                channels[i], channels[i+1],
                kernel_size=3, stride=2, padding=1, output_padding=1
            ))
            layers.append(nn.ReLU())

        layers.append(nn.ConvTranspose1d(
            channels[-1], action_dim,
            kernel_size=3, stride=2, padding=1, output_padding=1
        ))

        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        h = self.fc(z)
        h = h.view(z.size(0), -1, self.init_len)
        x = self.deconv(h)      # (B, action_dim, seq_len)
        return x.permute(0,2,1)


class VAE(nn.Module):
    def __init__(self, seq_len=60, action_dim=10,
                 enc_channels=[64,128,256],
                 dec_channels=[256,128,64],
                 latent_dim=64):
        super().__init__()

        self.encoder = ConvEncoder(
            seq_len=seq_len,
            action_dim=action_dim,
            channels=enc_channels,
            latent_dim=latent_dim
        )

        self.decoder = ConvDecoder(
            seq_len=seq_len,
            action_dim=action_dim,
            channels=dec_channels,
            latent_dim=latent_dim
        )

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        out = self.decoder(z)
        return out, mu, logvar

    @staticmethod
    def vae_loss(recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl, recon_loss, kl
