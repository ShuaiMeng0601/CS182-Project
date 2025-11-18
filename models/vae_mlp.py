# models/vae_mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------
#                  MLP Encoder
# --------------------------------------------------------
class MLPEncoder(nn.Module):
    def __init__(self, seq_len=60, action_dim=10, hidden_dim=256, layers=3, latent_dim=64):
        super().__init__()

        input_dim = seq_len * action_dim
        mlp_layers = []

        dim_in = input_dim
        dim_out = hidden_dim

        # MLP layers
        for _ in range(layers):
            mlp_layers.append(nn.Linear(dim_in, dim_out))
            mlp_layers.append(nn.ReLU())
            dim_in = dim_out

        self.mlp = nn.Sequential(*mlp_layers)

        # map to latent mean and logvar
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):   # x: (B, seq_len, action_dim)
        B = x.size(0)
        h = x.reshape(B, -1)         # flatten → (B, seq_len*action_dim)
        h = self.mlp(h)              # (B, hidden_dim)
        return self.fc_mu(h), self.fc_logvar(h)



# --------------------------------------------------------
#                  MLP Decoder
# --------------------------------------------------------
class MLPDecoder(nn.Module):
    def __init__(self, seq_len=60, action_dim=10, hidden_dim=256, layers=3, latent_dim=64):
        super().__init__()

        output_dim = seq_len * action_dim
        mlp_layers = []

        dim_in = latent_dim
        dim_out = hidden_dim

        for _ in range(layers):
            mlp_layers.append(nn.Linear(dim_in, dim_out))
            mlp_layers.append(nn.ReLU())
            dim_in = dim_out

        # final output
        mlp_layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*mlp_layers)
        self.seq_len = seq_len
        self.action_dim = action_dim

    def forward(self, z):  # z: (B, latent_dim)
        h = self.mlp(z)   # (B, seq_len*action_dim)
        return h.reshape(z.size(0), self.seq_len, self.action_dim)



# --------------------------------------------------------
#                     VAE (MLP)
# --------------------------------------------------------
class MLPVAE(nn.Module):
    def __init__(self, seq_len=60, action_dim=10,
                 hidden_dim=256,
                 layers=3,
                 latent_dim=64):
        super().__init__()

        self.encoder = MLPEncoder(
            seq_len=seq_len,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            latent_dim=latent_dim
        )

        self.decoder = MLPDecoder(
            seq_len=seq_len,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            latent_dim=latent_dim
        )

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std     # reparameterization
        out = self.decoder(z)
        return out, mu, logvar

    @staticmethod
    def vae_loss(recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl, recon_loss, kl
