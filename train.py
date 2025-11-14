# train.py
import yaml
import torch
from torch.utils.data import DataLoader
from dataset.seq_dataset import SequenceDataset
from models.vae import VAE
import argparse

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(cfg_path):
    cfg = load_config(cfg_path)

    ds = SequenceDataset(cfg["data"]["dataset_path"])
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True)

    model = VAE(
        input_dim=cfg["model"]["input_dim"],
        latent_dim=cfg["model"]["latent_dim"],
        enc_ch=cfg["model"]["encoder_channels"],
        dec_ch=cfg["model"]["decoder_channels"]
    ).to(cfg["train"]["device"])

    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    device = cfg["train"]["device"]

    for epoch in range(cfg["train"]["epochs"]):
        for x in dl:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, rc, kl = model.vae_loss(recon, x, mu, logvar)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()

        print(f"[Epoch {epoch}] Loss={loss.item():.4f}  Recon={rc.item():.4f}  KL={kl.item():.4f}")

    torch.save(model.state_dict(), "vae.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vae.yaml")
    args = parser.parse_args()
    main(args.config)
