"""
LCM — Image Codec

Converts images to discrete token ids and back.
Three backend options:
  1. vqgan       — pre-trained taming-transformers VQGAN (best quality)
  2. simple_vq   — lightweight custom VQ-VAE (trainable, fits T4)
  3. random_codebook — deterministic hash-based codec (no training needed, for testing)

Data flow:
  encode:  PIL.Image → resize → encoder → quantize → token ids [h×w]
  decode:  token ids [h×w] → codebook lookup → decoder → PIL.Image

Usage:
  from image_codec import ImageCodec

  codec = ImageCodec(codec_type='simple_vq', resolution=256, codebook_size=1024)
  token_ids = codec.encode("photo.jpg")        # → List[int]
  image     = codec.decode(token_ids, (16,16))  # → PIL.Image
"""

import os
import math
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ═══════════════════════════════════════════════════════════
# BASE CLASS
# ═══════════════════════════════════════════════════════════

class ImageCodec:
    """
    Unified interface for image ↔ token conversion.

    Parameters
    ----------
    codec_type     : 'vqgan' | 'simple_vq' | 'random_codebook'
    resolution     : square image size (default 256)
    codebook_size  : number of discrete codes (default 1024)
    latent_dim     : embedding dim per code (default 256)
    checkpoint_path: path to saved weights (optional)
    device         : 'cuda' or 'cpu'
    """

    def __init__(
        self,
        codec_type: str = "simple_vq",
        resolution: int = 256,
        codebook_size: int = 1024,
        latent_dim: int = 256,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.codec_type    = codec_type
        self.resolution    = resolution
        self.codebook_size = codebook_size
        self.latent_dim    = latent_dim
        self.device        = device

        if codec_type == "vqgan":
            self.backend = VQGANBackend(
                resolution, codebook_size, latent_dim, device
            )
        elif codec_type == "simple_vq":
            self.backend = SimpleVQBackend(
                resolution, codebook_size, latent_dim, device
            )
        elif codec_type == "random_codebook":
            self.backend = RandomCodebookBackend(
                resolution, codebook_size, latent_dim, device
            )
        else:
            raise ValueError(f"Unknown codec_type: {codec_type}")

        # load pre-trained weights if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load(checkpoint_path)

        print(
            f"ImageCodec [{codec_type}]: "
            f"res={resolution}, codes={codebook_size}, "
            f"latent={latent_dim}, grid={self.grid_shape}"
        )

    @property
    def grid_shape(self) -> Tuple[int, int]:
        """Spatial grid of token ids (h, w)."""
        return self.backend.grid_shape

    @property
    def tokens_per_image(self) -> int:
        """Number of tokens per image."""
        h, w = self.grid_shape
        return h * w

    def encode(self, image: Union[str, Image.Image, np.ndarray]) -> List[int]:
        """
        Image → flat list of discrete token ids.

        Parameters
        ----------
        image : file path, PIL Image, or numpy array [H,W,3] uint8

        Returns
        -------
        List[int] of length h×w, values in [0, codebook_size)
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # resize to target resolution
        image = image.resize(
            (self.resolution, self.resolution), Image.LANCZOS
        )
        return self.backend.encode(image)

    def decode(
        self,
        token_ids: List[int],
        grid_shape: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        """
        Token ids → PIL Image.

        Parameters
        ----------
        token_ids  : List[int] of length h×w
        grid_shape : (h, w) — if None, uses self.grid_shape

        Returns
        -------
        PIL.Image [resolution × resolution × 3]
        """
        if grid_shape is None:
            grid_shape = self.grid_shape
        return self.backend.decode(token_ids, grid_shape)

    def save(self, path: str):
        """Save codec weights to disk."""
        self.backend.save(path)

    def load(self, path: str):
        """Load codec weights from disk."""
        self.backend.load(path, self.device)


# ═══════════════════════════════════════════════════════════
# BACKEND 1: VQGAN  (taming-transformers)
# ═══════════════════════════════════════════════════════════

class VQGANBackend:
    """
    Wrapper around pre-trained taming-transformers VQGAN.

    Requires: pip install taming-transformers-rom1504
    Pre-trained checkpoints:
      - f8  → 256×256 image → 32×32 grid = 1024 tokens
      - f16 → 256×256 image → 16×16 grid = 256 tokens
      - f16 → 512×512 image → 32×32 grid = 1024 tokens

    If taming-transformers is not installed, falls back to SimpleVQBackend.
    """

    def __init__(self, resolution, codebook_size, latent_dim, device):
        self.resolution = resolution
        self.device     = device

        # try to load a pre-trained VQGAN
        self.model = None
        self.grid = (resolution // 16, resolution // 16)  # default f16

        try:
            self._load_taming(codebook_size)
        except ImportError:
            print(
                "WARNING: taming-transformers not installed. "
                "Falling back to SimpleVQBackend."
            )
            self._fallback = SimpleVQBackend(
                resolution, codebook_size, latent_dim, device
            )
            self.model = "fallback"

    def _load_taming(self, codebook_size):
        """Attempt to load taming-transformers VQGAN."""
        from taming.models.vqgan import VQModel

        # try common config/checkpoint pairs
        configs = [
            # (config_url, ckpt_url, expected_codebook)
        ]
        # for local checkpoints:
        ckpt_dir = Path("checkpoints/vqgan")
        if ckpt_dir.exists():
            ckpt = list(ckpt_dir.glob("*.ckpt"))[0]
            config = list(ckpt_dir.glob("*.yaml"))[0]
            # load would go here
            pass

        if self.model is None:
            raise ImportError("No VQGAN checkpoint found")

    @property
    def grid_shape(self):
        if self.model == "fallback":
            return self._fallback.grid_shape
        return self.grid

    def encode(self, image: Image.Image) -> List[int]:
        if self.model == "fallback":
            return self._fallback.encode(image)

        # real VQGAN encode path
        import torchvision.transforms as T
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        x = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            quant, emb_loss, info = self.model.encode(x)
            # info[2] contains codebook indices
            indices = info[2].view(-1).cpu().tolist()
        return indices

    def decode(self, token_ids: List[int], grid_shape) -> Image.Image:
        if self.model == "fallback":
            return self._fallback.decode(token_ids, grid_shape)

        indices = torch.tensor(token_ids, dtype=torch.long).to(self.device)
        indices = indices.view(1, grid_shape[0], grid_shape[1])
        with torch.no_grad():
            quant = self.model.quantize.embedding(indices)
            quant = quant.permute(0, 3, 1, 2)  # [B,C,H,W]
            x = self.model.decode(quant)
        x = (x.clamp(-1, 1) + 1) * 127.5
        x = x.byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(x)

    def save(self, path):
        if self.model == "fallback":
            return self._fallback.save(path)
        torch.save(self.model.state_dict(), path)

    def load(self, path, device):
        if self.model == "fallback":
            return self._fallback.load(path, device)
        self.model.load_state_dict(torch.load(path, map_location=device))


# ═══════════════════════════════════════════════════════════
# BACKEND 2: SIMPLE VQ-VAE  (lightweight, trainable)
# ═══════════════════════════════════════════════════════════

class SimpleVQEncoder(nn.Module):
    """
    Conv encoder: image [B,3,H,W] → latent [B, latent_dim, h, w]

    Architecture:
      4 conv layers, stride 2 each → downsample 16×
      256×256 → 16×16 spatial
    """

    def __init__(self, in_ch=3, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            # 256 → 128
            nn.Conv2d(in_ch, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            # 128 → 64
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            # 64 → 32
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            # 32 → 16
            nn.Conv2d(256, latent_dim, 4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.net(x)  # [B, latent_dim, H/16, W/16]


class SimpleVQDecoder(nn.Module):
    """
    Conv decoder: latent [B, latent_dim, h, w] → image [B, 3, H, W]

    Architecture:
      4 transposed conv layers, stride 2 each → upsample 16×
      16×16 → 256×256 spatial
    """

    def __init__(self, latent_dim=256, out_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            # 16 → 32
            nn.ConvTranspose2d(latent_dim, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            # 32 → 64
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            # 64 → 128
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            # 128 → 256
            nn.ConvTranspose2d(64, out_ch, 4, stride=2, padding=1),
            nn.Tanh(),  # output in [-1, 1]
        )

    def forward(self, z):
        return self.net(z)  # [B, 3, H, W]


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with EMA codebook update.

    input  : [B, D, h, w]
    output : quantized [B, D, h, w], indices [B, h, w]

    Codebook: [codebook_size, D]
    """

    def __init__(self, codebook_size: int, latent_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.codebook_size  = codebook_size
        self.latent_dim     = latent_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(codebook_size, latent_dim)
        nn.init.uniform_(
            self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size
        )

        # EMA tracking (updated during training only)
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embedding_sum", self.embedding.weight.clone())
        self.register_buffer("initted", torch.tensor(False))
        self.decay = 0.99

    def forward(self, z: torch.Tensor):
        """
        Parameters
        ----------
        z : [B, D, h, w]  encoder output

        Returns
        -------
        z_q      : [B, D, h, w]   quantized
        indices  : [B, h, w]       codebook indices
        loss     : scalar          commitment + embedding loss
        """
        B, D, h, w = z.shape

        # reshape to [B*h*w, D]
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)  # [N, D]

        # distances to codebook: ||z - e||²
        # = ||z||² + ||e||² - 2*z·e
        dist = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2.0 * z_flat @ self.embedding.weight.t()
        )  # [N, codebook_size]

        # find nearest codebook entry
        indices = dist.argmin(dim=1)  # [N]
        z_q = self.embedding(indices)  # [N, D]

        # EMA update (training only)
        if self.training:
            one_hot = F.one_hot(indices, self.codebook_size).float()
            cluster_size = one_hot.sum(dim=0)
            embedding_sum = one_hot.t() @ z_flat

            if not self.initted:
                self.ema_cluster_size.copy_(cluster_size)
                self.ema_embedding_sum.copy_(embedding_sum)
                self.initted.fill_(True)
            else:
                self.ema_cluster_size.mul_(self.decay).add_(
                    cluster_size, alpha=1.0 - self.decay
                )
                self.ema_embedding_sum.mul_(self.decay).add_(
                    embedding_sum, alpha=1.0 - self.decay
                )

            # Laplace smoothing
            n = self.ema_cluster_size.sum()
            cluster_size_smoothed = (
                (self.ema_cluster_size + 1e-5)
                / (n + self.codebook_size * 1e-5)
                * n
            )
            self.embedding.weight.data.copy_(
                self.ema_embedding_sum / cluster_size_smoothed.unsqueeze(1)
            )

        # straight-through estimator
        z_q_st = z_flat + (z_q - z_flat).detach()

        # losses
        embedding_loss = F.mse_loss(z_q.detach(), z_flat)
        commitment_loss = F.mse_loss(z_q_st, z_flat.detach())
        loss = embedding_loss + self.commitment_cost * commitment_loss

        # reshape back
        z_q_out = z_q_st.reshape(B, h, w, D).permute(0, 3, 1, 2)  # [B,D,h,w]
        indices_out = indices.reshape(B, h, w)                       # [B,h,w]

        return z_q_out, indices_out, loss


class SimpleVQVAE(nn.Module):
    """
    Complete lightweight VQ-VAE for LCM.

    Image [B,3,256,256]
      → Encoder → [B, latent_dim, 16, 16]
      → VectorQuantizer → [B, latent_dim, 16, 16], indices [B, 16, 16]
      → Decoder → [B, 3, 256, 256]
    """

    def __init__(self, codebook_size=1024, latent_dim=256):
        super().__init__()
        self.encoder     = SimpleVQEncoder(3, latent_dim)
        self.quantizer   = VectorQuantizer(codebook_size, latent_dim)
        self.decoder     = SimpleVQDecoder(latent_dim, 3)
        self.latent_dim  = latent_dim

    def encode(self, x):
        """x: [B,3,H,W] → z_q: [B,D,h,w], indices: [B,h,w], vq_loss: scalar"""
        z = self.encoder(x.quantizer(z)
        return z_q, indices, vq_loss

    def decode(self, z_q):
        """z_q:)
        z_q, indices, vq_loss = self [B,D,h,w] → x_recon: [B,3,H,W]"""
        return self.decoder(z_q)

    def forward(self, x):
        """Full forward: encode → quantize → decode."""
        z_q, indices, vq_loss = self.encode(x)
        x_recon = self.decode(z_q)
        return x_recon, indices, vq_loss


class SimpleVQBackend:
    """
    Backend using SimpleVQVAE.
    Can be trained standalone before LCM training,
    or loaded from a pre-trained checkpoint.
    """

    def __init__(self, resolution, codebook_size, latent_dim, device):
        self.resolution   = resolution
        self.codebook_size = codebook_size
        self.latent_dim   = latent_dim
        self.device       = device
        self.downsample   = 16  # 4 conv layers, stride 2

        self.grid = (resolution // self.downsample, resolution // self.downsample)

        self.model = SimpleVQVAE(codebook_size, latent_dim).to(device)
        self.model.eval()

        # preprocessing: [0,255] → [-1,1]
        self.transform = lambda img: (
            torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 127.5 - 1.0
        )

    @property
    def grid_shape(self):
        return self.grid

    @torch.no_grad()
    def encode(self, image: Image.Image) -> List[int]:
        x = self.transform(image).unsqueeze(0).to(self.device)
        _, indices, _ = self.model.encode(x)
        return indices.view(-1).cpu().tolist()

    @torch.no_grad()
    def decode(self, token_ids: List[int], grid_shape) -> Image.Image:
        h, w = grid_shape
        indices = torch.tensor(token_ids, dtype=torch.long).to(self.device)
        indices = indices.view(1, h, w)

        # codebook lookup
        z_q = self.model.quantizer.embedding(indices)    # [1, h, w, D]
        z_q = z_q.permute(0, 3, 1, 2)                   # [1, D, h, w]
        x_recon = self.model.decode(z_q)                  # [1, 3, H, W]

        x = (x_recon.clamp(-1, 1) + 1) * 127.5
        x = x.byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(x)

    def save(self, path):
        torch.save({
            'model_state': self.model.state_dict(),
            'codebook_size': self.codebook_size,
            'latent_dim': self.latent_dim,
            'resolution': self.resolution,
        }, path)
        print(f"Saved SimpleVQVAE to {path}")

    def load(self, path, device):
        ckpt = torch.load(path, map_location=device)
        self.model.load_state_dict(ckpt['model_state'])
        self  (deterministic, no training)
# ═══════════════════════════════════════════════════════════

class RandomCodebookBackend:
    """
    Deterministic hash-based codec for testing without════════════════════════.model.eval()
        print(f"Loaded SimpleVQVAE from {path}")


# ═══════════════════════════════════
# BACKEND 3: RANDOM CODEBOOK training.

    encode: divide image into patches → nearest neighbor in random codebook
    decode: codebook lookup → reshape to image patches → reassemble

    The codebook is generated from a fixed seed so it's
    reproducible across runs without saving weights.
    """

    def __init__(self, resolution, codebook_size, latent_dim, device):
        self.resolution    = resolution
        self.codebook_size = codebook_size
        self.latent_dim    = latent_dim
        self.device        = device
        self.downsample    = 16
        self.grid = (resolution // self.downsample, resolution // self.downsample)
        self.patch_size    = resolution // self.grid[0]  # 16 for 256/16

        # fixed random codebook
        rng = torch.Generator().manual_seed(42)
        self.codebook = torch.randn(
            codebook_size, 3 * self.patch_size * self.patch_size,
            generator=rng,
        )
        # normalize
        self.codebook = F.normalize(self.codebook, dim=1)

    @property
    def grid_shape(self):
        return self.grid

    def encode(self, image: Image.Image) -> List[int]:
        arr = np.array(image).astype(np.float32) / 255.0  # [H,W,3]
        h, w = self.grid
        ps = self.patch_size
        codes = []
        for i in range(h):
            for j in range(w):
                patch = arr[i * ps : (i + 1) * ps, j * ps : (j + 1) * ps]
                patch_flat = torch.from_numpy(patch.reshape(1, -1))
                patch_flat = F.normalize(patch_flat, dim=1)
                # nearest codebook entry
                sim = patch_flat @ self.codebook.t()  # [1, codebook_size]
                code = sim.argmax(dim=1).item()
                codes.append(code)
        return codes

    def decode(self, token_ids: List[int], grid_shape) -> Image.Image:
        h, w = grid_shape
        ps = self.patch_size
        img = np.zeros((h * ps, w * ps, 3), dtype=np.float32)
        idx = 0
        for i in range(h):
            for j in range(w):
                code = token_ids[idx]
                patch = self.codebook[code].numpy().reshape(ps, ps, 3)
                # res.save({'cale from normalized to [0,1]
                patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
                img[i * ps : (i + 1) * ps, j * ps : (j + 1)codebook': self.codebook}, path)

    def load(self, path, device):
        ckpt = torch.load(path, map_location=device)
        self.codebook = ckpt['codebook']


# ═══════════════════════════════════════════════════════════
# STANDALONE VQ-VAE TRAINING SCRIPT
# ═══════════════════════════════════════════════════════════

def train_simple_vqvae(
    image_dir: str,
    save_path: str,
    resolution: int = 256,
    codebook_size: int * ps] = patch
                idx += 1
        img = (img * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img)

    def save(self, path):
        torch = 1024,
    latent_dim: int = 256,
    batch_size: int = 16,
    max_steps: int = 50000,
    lr: float = 3e-4,
    device: str = "cuda",
):
    """
    Train a SimpleVQVAE on a directory of images.
    Run this BEFORE LCM training.

    Usage:
      python image_codec.py --train --image_dir data/images --save checkpoints/vq.pt
    """
    from torch.utils.data import Dataset, DataLoader as DL
    import torchvision.transforms as T

    class ImageFolder(Dataset):
        def __init__(self, root, res):
            self.paths = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
                self.paths.extend(Path(root).rglob(ext))
            self.paths = sorted(self.paths)
            self.transform = T.Compose([
                T.Resize((res, res)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img)

    dataset = ImageFolder(image_dir, resolution)
    loader  = DL(dataset, batch_size=batch_size, shuffle=True,
                 num_workers=4, pin_memory=True, drop_last=True)
    print(f"VQ-VAE training: {len(dataset)} images, {len(loader)} batches/epoch")

    model = SimpleVQVAE(codebook_size, latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for step in range(max_steps):
        for batch in loader:
            x = batch.to(device)
            x_recon, indices, vq_loss = model(x)
            recon_loss = F.mse_loss(x_recon, x)
            perceptual_loss = _lpips_simple(x_recon, x)
            total_loss = recon_loss + vq_loss + 0.1 * perceptual_loss

            opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 100 == 0:
                usage = indices.unique().numel()
                print(
                    f"step {step:>6d} | "
                    f"recon {recon_loss:.4f} | "
                    f"vq {vq_loss:.4f} | "
                    f"usage {usage}/{codebook_size}"
                )
            if step % 5000 == 0 and step > 0:
                backend = SimpleVQBackend(resolution, codebook_size, latent_dim, device)
                backend.model = model
                backend.save(save_path)

            step += 1
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    # final save
    backend = SimpleVQBackend(resolution, codebook_size, latent_dim, device)
    backend.model = model
    backend.save(save_path)
    print(f"VQ-VAE training complete. Saved to {save_path}")


def _lpips_simple(x, y):
    """
    Simple perceptual loss (L1 in pixel space + frequency).
    A real implementation would use a pre-trained VGG,
    but this works without extra dependencies.
    """
    # L1 pixel loss
    l1 = F.l1_loss(x, y)
    # high-frequency emphasis (penalize blurry reconstructions)
    dx_x = x[:, :, :, 1:] - x[:, :, :, :-1]
    dx_y = y[:, :, :, 1:] - y[:, :, :, :-1]
    dy_x = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy_y = y[:, :, 1:, :] - y[:, :, :-1, :]
    freq = F.l1_loss(dx_x, dx_y) + F.l1_loss(dy_x, dy_y)
    return l1 + 0.5 * freq


# ═══════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LCM Image Codec")
    sub = parser.add_subparsers(dest="command")

    # ── train ──
    p_train = sub.add_parser("train", help="Train SimpleVQVAE")
    p_train.add_argument("--image_dir",  type=str, required=True)
    p_train.add_argument("--save",       type=str, default="checkpoints/vq.pt")
    p_train.add_argument("--resolution", type=int, default=256)
    p_train.add_argument("--codebook",   type=int, default=1024)
    p_train.add_argument("--latent_dim", type=int, default=256)
    p_train.add_argument("--batch_size", type=int, default=16)
    p_train.add_argument("--max_steps",  type=int, default=50000)
    p_train.add_argument("--lr",         type=float, default=3e-4)
    p_train.add_argument("--device",     type=str, default="cuda")

    # ── encode ──
    p_enc = sub.add_parser("encode", help="Encode an image to tokens")
    p_enc.add_argument("--image",      type=str, required=True)
    p_enc.add_argument("--codec",      type=str, default="simple_vq")
    p_enc.add_argument("--checkpoint", type=str, default=None)
    p_enc.add_argument("--resolution", type=int, default=256)
    p_enc.add_argument("--codebook",   type=int, default=1024)
    p_enc.add_argument("--device",     type=str, default="cuda")

    # ── decode ──
    p_dec = sub.add_parser("decode", help="Decode tokens to image")
    p_dec.add_argument("--tokens",     type=str, required=True,
                       help="Comma-separated token ids or path to .npy")
    p_dec.add_argument("--output",     type=str, default="decoded.png")
    p_dec.add_argument("--codec",      type=str, default="simple_vq")
    p_dec.add_argument("--checkpoint", type=str, default=None)
    p_dec.add_argument("--resolution", type=int, default=256)
    p_dec.add_argument("--codebook",   type=int, default=1024)
    p_dec.add_argument("--device",     type=str, default="cuda")

    # ── roundtrip ──
    p_rt = sub.add_parser("roundtrip", help="Encode → decode → save (test)")
    p_rt.add_argument("--image",       type=str, required=True)
    p_rt.add_argument("--output",      type=str, default="roundtrip.png")
    p_rt.add_argument("--codec",       type=str, default="simple_vq")
    p_rt.add_argument("--checkpoint",  type=str, default=None)
    p_rt.add_argument("--resolution",  type=int, default=256)
    p_rt.add_argument("--codebook",    type=int, default=1024)
    p_rt.add_argument("--device",      type=str, default="cuda")

    args = parser.parse_args()

    if args.command == "train":
        train_simple_vqvae(
            image_dir=args.image_dir,
            save_path=args.save,
            resolution=args.resolution,
            codebook_size=args.codebook,
            latent_dim=args.latent_dim,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            lr=args.lr,
            device=args.device,
        )

    elif args.command == "encode":
        codec = ImageCodec(
            codec_type=args.codec,
            resolution=args.resolution,
            codebook_size=args.codebook,
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
        tokens = codec.encode(args.image)
        print(f"Tokens ({len(tokens)}): {tokens[:20]}...")
        np.save(args.image + ".tokens.npy", np.array(tokens, dtype=np.int32))
        print(f"Saved to {args.image}.tokens.npy")

    elif args.command == "decode":
        codec = ImageCodec(
            codec_type=args.codec,
            resolution=args.resolution,
            codebook_size=args.codebook,
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
        if os.path.exists(args.tokens):
            tokens = np.load(args.tokens`

---

## Usage Examples

```bash
# 1. Train VQ-VAE on).tolist()
        else:
            tokens = [int(t.strip()) for t in args.tokens.split(",")]
        h, w = codec.grid_shape
        img = codec.decode(tokens, (h, w))
        img.save(args.output)
        print(f"Saved decoded image to {args.output}")

    elif args.command == "roundtrip":
        codec = ImageCodec(
            codec_type=args.codec,
            resolution=args.resolution,
            codebook_size=args.codebook,
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
        print(f"Encoding {args.image}...")
        tokens = codec.encode(args.image)
        print(f"  → {len(tokens)} tokens, "
              f"unique: {len(set(tokens))}/{codec.codebook_size}")
        print(f"Decoding...")
        img = codec.decode(tokens, codec.grid_shape)
        img.save(args.output)
        print(f"Roundtrip saved to {args.output}")

    else:
        parser.print_help()
`` your image data (do this FIRST)
python image_codec.py train \
    --image_dir data/images \
    --save checkpoints/vq.pt \
    --resolution 256 \
    --codebook 1024 \
    --max_steps 50000

# 2. Test roundtrip quality
python image_codec.py roundtrip \
    --image test.jpg \
    --output test_roundtrip.png \
    --codec simple_vq \
    --checkpoint checkpoints/vq.pt

# 3. Encode an image to tokens (for data preparation)
python image_codec.py encode \
    --image photo.jpg \
    --checkpoint checkpoints/vq.pt

# 4. In multimodal_loader.py (automatic)
#    ImageTokenizer calls ImageCodec internally