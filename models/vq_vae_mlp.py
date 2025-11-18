# models/vqvae_mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------
#      Vector Quantizer with EMA (same as CNN version)
# --------------------------------------------------------
class VectorQuantizerEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps

        embed = torch.randn(n_embeddings, embedding_dim)
        self.register_buffer("embedding", embed)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", embed.clone())

    def forward(self, z_e):  # z_e: (B, T, D)
        B, T, D = z_e.shape
        flat = z_e.reshape(-1, D)  # (B*T, D)

        # squared Euclidean distance
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.t()
            + self.embedding.pow(2).sum(1)
        )  # (B*T, n_embeddings)

        indices = torch.argmin(dist, dim=1)  # (B*T,)
        z_q = F.embedding(indices, self.embedding)  # (B*T, D)
        z_q = z_q.view(B, T, D)
        indices = indices.view(B, T)

        # EMA updates
        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(indices.view(-1), self.n_embeddings).float()

                # update ema count
                new_count = encodings.sum(0)
                self.ema_count.mul_(self.decay).add_(new_count, alpha=1 - self.decay)

                # normalize
                total = self.ema_count.sum()
                self.ema_count = (self.ema_count + self.eps) / \
                                 (total + self.n_embeddings * self.eps) * total

                # update embedding
                new_weight = encodings.t() @ flat
                self.ema_weight.mul_(self.decay).add_(new_weight, alpha=1 - self.decay)

                # final normalized embedding
                self.embedding.copy_(self.ema_weight / self.ema_count.unsqueeze(1))

        # loss
        codebook_loss = F.mse_loss(z_e.detach(), z_q)
        commitment_loss = F.mse_loss(z_e, z_q.detach())

        # straight-through estimator
        z_q = z_e + (z_q - z_e).detach()

        return z_q, indices, codebook_loss, commitment_loss



# --------------------------------------------------------
#              MLP Encoder (frame-wise)
# --------------------------------------------------------
class MLPEncoder(nn.Module):
    def __init__(self, action_dim=10, hidden_dim=256, latent_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):  # x: (B, T, action_dim)
        return self.mlp(x)  # (B, T, latent_dim)



# --------------------------------------------------------
#              MLP Decoder (frame-wise)
# --------------------------------------------------------
class MLPDecoder(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256, action_dim=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, z_q):  # z_q: (B, T, latent_dim)
        return self.mlp(z_q)  # (B, T, action_dim)



# --------------------------------------------------------
#                     MLP VQ-VAE
# --------------------------------------------------------
class VQVAE_MLP(nn.Module):
    def __init__(
        self,
        action_dim=10,
        latent_dim=128,
        hidden_dim=256,
        n_embeddings=512,
    ):
        super().__init__()

        self.encoder = MLPEncoder(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )

        self.quantizer = VectorQuantizerEMA(
            n_embeddings=n_embeddings,
            embedding_dim=latent_dim
        )

        self.decoder = MLPDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim
        )

    def forward(self, x):  # x: (B, T, action_dim)
        z_e = self.encoder(x)
        z_q, indices, codebook_loss, commitment_loss = self.quantizer(z_e)
        out = self.decoder(z_q)
        return out, codebook_loss, commitment_loss, indices

    @staticmethod
    def vqvae_loss(recon_x, x, codebook_loss, commitment_loss, beta=0.25):
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        total = recon_loss + codebook_loss + beta * commitment_loss
        return total, recon_loss, codebook_loss, commitment_loss
