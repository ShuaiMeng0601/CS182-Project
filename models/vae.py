# models/vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    def __init__(self, input_dim=510, channels=[64,128,256], latent_dim=64):
        super().__init__()
        self.input_dim = input_dim

        layers = []
        in_ch = 1
        length = input_dim
        for ch in channels:
            layers.append(nn.Conv1d(in_ch, ch, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU())
            in_ch = ch
            length //= 2
        self.conv = nn.Sequential(*layers)
        self.final_len = length

        self.fc_mu = nn.Linear(channels[-1] * length, latent_dim)
        self.fc_logvar = nn.Linear(channels[-1] * length, latent_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class ConvDecoder(nn.Module):
    def __init__(self, output_dim=510, channels=[256,128,64], latent_dim=64):
        super().__init__()
        self.output_dim = output_dim

        self.init_len = output_dim // (2 ** len(channels))
        self.fc = nn.Linear(latent_dim, channels[0] * self.init_len)

        layers = []
        in_chs = channels
        for i in range(len(channels)-1):
            layers.append(nn.ConvTranspose1d(
                in_chs[i], in_chs[i+1], kernel_size=4, stride=2, padding=1
            ))
            layers.append(nn.ReLU())
        layers.append(nn.ConvTranspose1d(channels[-1], 1, kernel_size=4, stride=2, padding=1))
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        h = self.fc(z)
        h = h.view(z.size(0), -1, self.init_len)
        x = self.deconv(h)
        return x.squeeze(1)[:, :self.output_dim]


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, enc_ch, dec_ch):
        super().__init__()
        self.encoder = ConvEncoder(input_dim, enc_ch, latent_dim)
        self.decoder = ConvDecoder(input_dim, dec_ch, latent_dim)

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
