# ============================================================================
# Tiny VAE Training Script
# ============================================================================
# Author: AbstractPhil
# Description: Train a miniature VAE on FashionMNIST or MNIST.

# License: MIT


# ============================================================================
# CELL 1: Train and Save VAE Encoder
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


class TinyVAE(nn.Module):
    """Miniature VAE: 28x28 → 7x7x4 latent (like Stable Diffusion compression)"""

    def __init__(self, latent_channels=16):
        super().__init__()

        # Encoder: 28x28 → 7x7
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 7x7
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        # Latent space projections
        self.fc_mu = nn.Conv2d(64, latent_channels, 1)
        self.fc_logvar = nn.Conv2d(64, latent_channels, 1)

        # Decoder: 7x7 → 28x28
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # 28x28
            nn.Tanh()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta=0.1):
    """VAE loss: reconstruction + KL divergence"""
    recon_loss = F.mse_loss(recon, x, reduction='sum') / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_vae_epoch(vae, loader, optimizer, device):
    vae.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    pbar = tqdm(loader, desc='Training VAE')
    for images, _ in pbar:
        images = images.to(device, non_blocking=True)

        optimizer.zero_grad()
        recon, mu, logvar = vae(images)
        loss, recon_loss, kl_loss = vae_loss(recon, images, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}'
        })

    n = len(loader)
    return total_loss / n, total_recon / n, total_kl / n


def visualize_vae_reconstructions(vae, loader, device, num_samples=10):
    vae.eval()
    images, _ = next(iter(loader))
    images = images[:num_samples].to(device)

    with torch.inference_mode():
        recon, mu, _ = vae(images)

    images = images.cpu()
    recon = recon.cpu()

    fig, axes = plt.subplots(2, num_samples, figsize=(20, 4))
    for idx in range(num_samples):
        # Original
        axes[0, idx].imshow(images[idx].squeeze(), cmap='gray')
        axes[0, idx].axis('off')
        if idx == 0:
            axes[0, idx].set_title('Original', fontsize=10)

        # Reconstruction
        axes[1, idx].imshow(recon[idx].squeeze(), cmap='gray')
        axes[1, idx].axis('off')
        if idx == 0:
            axes[1, idx].set_title('Reconstructed', fontsize=10)

    plt.tight_layout()
    plt.show()


def main_vae_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training VAE on {device}")

    # Data (use same normalization as before)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    # Change to FashionMNIST or MNIST as needed
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Create VAE
    vae = TinyVAE(latent_channels=4).to(device)
    total_params = sum(p.numel() for p in vae.parameters())
    print(f"VAE Parameters: {total_params:,} ({total_params * 4 / 1024:.2f} KB)\n")

    # Optimizer
    optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    # Training
    print("Training VAE for 15 epochs...\n")
    history = {'train_loss': [], 'recon_loss': [], 'kl_loss': []}

    for epoch in range(15):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/15")
        print(f"{'=' * 60}")

        loss, recon, kl = train_vae_epoch(vae, train_loader, optimizer, device)
        scheduler.step()

        history['train_loss'].append(loss)
        history['recon_loss'].append(recon)
        history['kl_loss'].append(kl)

        print(f"\nTotal Loss: {loss:.4f} | Recon: {recon:.4f} | KL: {kl:.4f}")

    # Visualize results
    print("\n" + "=" * 60)
    print("VAE Training Complete!")
    print("=" * 60)
    visualize_vae_reconstructions(vae, test_loader, device)

    # Save model
    save_path = 'fashion_vae_encoder.pt'
    torch.save({
        'model_state_dict': vae.state_dict(),
        'latent_channels': 16,
        'architecture': 'TinyVAE',
    }, save_path)
    print(f"\n✓ VAE saved to: {save_path}")

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history['train_loss'], label='Total Loss')
    ax1.set_title('VAE Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['recon_loss'], label='Reconstruction')
    ax2.plot(history['kl_loss'], label='KL Divergence')
    ax2.set_title('VAE Loss Components')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    return vae


# Begin training with this.
#if __name__ == "__main__":
#    vae = main_vae_training()