
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np

# -------- Positional Encoding --------
class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs, include_input=True):
        super().__init__()
        self.include_input = include_input
        self.freq_bands = 2. ** torch.linspace(0, num_freqs - 1, num_freqs)

    def forward(self, x):
        out = [x] if self.include_input else []
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)

# -------- NeRF MLP --------
class NeRF(nn.Module):
    def __init__(self, pos_dim=10, dir_dim=4):
        super().__init__()
        self.pos_enc = PositionalEncoding(pos_dim)
        self.dir_enc = PositionalEncoding(dir_dim)

        self.fc_pos = nn.Sequential(
            nn.Linear(pos_dim * 6 + 3, 256), nn.ReLU(),
            *[nn.Sequential(nn.Linear(256, 256), nn.ReLU()) for _ in range(4)]
        )
        self.fc_sigma = nn.Linear(256, 1)
        self.fc_feat = nn.Linear(256, 256)

        self.fc_dir = nn.Sequential(
            nn.Linear(dir_dim * 6 + 3 + 256, 128), nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x, d):
        x_encoded = self.pos_enc(x)
        d_encoded = self.dir_enc(d)
        h = self.fc_pos(x_encoded)
        sigma = self.fc_sigma(h)
        feat = self.fc_feat(h)
        h_rgb = torch.cat([feat, d_encoded], dim=-1)
        rgb = self.fc_dir(h_rgb)
        return torch.cat([rgb, sigma], dim=-1)

# -------- Volume Rendering (Stable) --------
def volume_rendering(rgb_sigma, z_vals):
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], -1)
    rgb = torch.sigmoid(rgb_sigma[..., :3])  # restrict RGB to (0, 1)
    sigma = rgb_sigma[..., 3]
    sigma = torch.clamp(sigma, 0.0, 1e3)
    alpha = 1. - torch.exp(-sigma * dists)
    weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], -1), -1)[..., :-1]
    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    return rgb_map

# -------- Training Loop Using Image --------
def train_nerf():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load target image
    image = imageio.imread("target_image_32x32.png") / 255.0
    target_rgb = torch.tensor(image.reshape(-1, 3), dtype=torch.float32).to(device)

    # Embed (i, j) positions as (x, y, z=0) into 3D world
    H, W = 32, 32
    i, j = torch.meshgrid(torch.linspace(-1, 1, W), torch.linspace(-1, 1, H), indexing='ij')
    pixel_coords = torch.stack([i, j, torch.zeros_like(i)], dim=-1).reshape(-1, 3).to(device)

    # Use these as the sample points (broadcast to N x 64 x 3)
    sample_pts = pixel_coords[..., None, :].expand(-1, 64, -1)
    z_vals = torch.linspace(0., 1., steps=64).expand(H * W, 64).to(device)

    # Dummy view direction (same for all)
    sample_dirs = torch.tensor([0, 0, 1.0], device=device).expand(H * W, 64, 3)

    nerf = NeRF().to(device)
    optimizer = torch.optim.Adam(nerf.parameters(), lr=5e-4)

    for i in range(1000):
        rgb_sigma = nerf(sample_pts.reshape(-1, 3), sample_dirs.reshape(-1, 3)).reshape(H * W, 64, 4)
        rgb_map = volume_rendering(rgb_sigma, z_vals)
        loss = F.mse_loss(rgb_map, target_rgb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Step {i}, Loss: {loss.item():.4f}")
            img = rgb_map.reshape(H, W, 3).detach().cpu().numpy()
            plt.imshow(img)
            plt.title(f"Rendered Step {i}")
            plt.axis('off')
            plt.savefig(f"render_step_{i}.png")
            plt.close()

if __name__ == "__main__":
    train_nerf()
