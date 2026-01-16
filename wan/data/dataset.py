# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Video dataset for Wan 2.2 training.

This module provides datasets for text-image-to-video (ti2v) training.
"""

import torch
from torch.utils.data import Dataset

__all__ = ['DummyVideoDataset', 'DummyLatentDataset']


class DummyVideoDataset(Dataset):
    """
    Dummy dataset for ti2v training that generates random video tensors,
    images (first frame), and text prompts.

    This dataset is useful for testing the training pipeline without
    requiring actual video data. It generates:
    - Random video tensors in pixel space [C=3, T, H, W]
    - Image tensor (first frame of video) [C=3, H, W]
    - Random text prompts

    The VAE and T5 encoding happens on-the-fly in the training loop.

    Args:
        num_samples: Number of samples in the dataset
        num_frames: Number of video frames (T)
        height: Video height in pixels
        width: Video width in pixels
        text_max_len: Maximum text sequence length
    """

    def __init__(
        self,
        num_samples: int = 1000,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        text_max_len: int = 512,
    ):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.text_max_len = text_max_len

        # Pre-generate random prompts (dummy text)
        self.prompts = [
            f"A video of random content {i}" for i in range(num_samples)
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dictionary with:
            - video: Random tensor [C=3, T, H, W] in range [-1, 1]
            - image: First frame of video [C=3, H, W] in range [-1, 1]
            - prompt: Text string for the video
        """
        # Generate random video tensor in pixel space [-1, 1]
        video = torch.randn(3, self.num_frames, self.height, self.width)
        video = video.clamp(-1, 1)

        # Image is the first frame of the video
        image = video[:, 0, :, :]  # [C=3, H, W]

        return {
            "video": video,
            "image": image,
            "prompt": self.prompts[idx],
        }


class DummyLatentDataset(Dataset):
    """
    Dummy dataset that generates pre-encoded latents and text embeddings.

    This is useful for testing when you want to skip VAE/T5 encoding.
    Latent dimensions follow Wan 2.2 VAE compression:
    - Temporal: T -> T/4 (after first frame)
    - Spatial: H -> H/8, W -> W/8
    - Channels: 3 -> 16

    Args:
        num_samples: Number of samples in the dataset
        num_frames: Number of video frames (T) - will be compressed to ~T/4
        height: Video height - will be compressed to H/8
        width: Video width - will be compressed to W/8
        text_embed_dim: Text embedding dimension (T5 outputs 4096)
        text_len: Text sequence length
    """

    def __init__(
        self,
        num_samples: int = 1000,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        text_embed_dim: int = 4096,
        text_len: int = 256,
    ):
        self.num_samples = num_samples

        # Compute latent dimensions (VAE compression)
        # Temporal: first frame + (remaining // 4)
        self.latent_frames = 1 + (num_frames - 1) // 4
        self.latent_height = height // 8
        self.latent_width = width // 8
        self.latent_channels = 16

        self.text_embed_dim = text_embed_dim
        self.text_len = text_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dictionary with:
            - video_latents: [C=16, F_latent, H_latent, W_latent]
            - image_latents: [C=16, 1, H_latent, W_latent] (first frame)
            - text_embeds: [L, D=4096]
        """
        # Generate random video latents
        video_latents = torch.randn(
            self.latent_channels,
            self.latent_frames,
            self.latent_height,
            self.latent_width
        )

        # Image latent is the first frame
        image_latents = video_latents[:, :1, :, :]  # [C, 1, H, W]

        # Generate random text embeddings
        # Vary the length slightly for realism
        actual_len = torch.randint(32, self.text_len + 1, (1,)).item()
        text_embeds = torch.randn(actual_len, self.text_embed_dim)

        return {
            "video_latents": video_latents,
            "image_latents": image_latents,
            "text_embeds": text_embeds,
        }
