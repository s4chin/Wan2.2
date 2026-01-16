# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Flow Matching loss for Wan 2.2 training.

Matches DiffSynth-Studio implementation with:
- Discrete timestep sampling via scheduler
- Wan-specific sigma schedule with shift=5
- Gaussian timestep-dependent loss weighting
- Configurable timestep boundaries
- TI2V support: fuse image latent into frame 0, use separated timesteps
"""

import torch
import torch.nn.functional as F

__all__ = ['FlowMatchScheduler', 'FlowMatchingLoss']


class FlowMatchScheduler:
    """
    Flow matching scheduler for Wan models.

    Handles sigma schedule, noise addition, and training weights.
    Matches DiffSynth-Studio's FlowMatchScheduler with Wan-specific settings.

    Args:
        num_train_timesteps: Number of training timesteps (default: 1000)
        shift: Timestep shift parameter for Wan (default: 5.0)
        sigma_min: Minimum sigma value (default: 0.0)
        sigma_max: Maximum sigma value (default: 1.0)
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        sigma_min: float = 0.0,
        sigma_max: float = 1.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Initialize with training timesteps
        self.set_timesteps(num_train_timesteps, training=True)

    def set_timesteps(
        self,
        num_inference_steps: int = 1000,
        denoising_strength: float = 1.0,
        training: bool = False,
        device: torch.device = None,
    ):
        """
        Set up timesteps and sigmas for training or inference.

        Args:
            num_inference_steps: Number of steps
            denoising_strength: Strength of denoising (1.0 = full)
            training: If True, compute training weights
            device: Device for tensors
        """
        # Compute sigma range
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength

        # Linear sigma schedule from sigma_start to sigma_min
        sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]

        # Apply Wan-specific shift transformation
        # sigma_shifted = shift * sigma / (1 + (shift - 1) * sigma)
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        # Convert sigmas to timesteps
        timesteps = sigmas * self.num_train_timesteps

        if device is not None:
            sigmas = sigmas.to(device)
            timesteps = timesteps.to(device)

        self.sigmas = sigmas
        self.timesteps = timesteps

        # Compute training weights if needed
        if training:
            self._set_training_weights(num_inference_steps)

    def _set_training_weights(self, num_steps: int):
        """
        Compute Gaussian training weights emphasizing middle timesteps.

        Weight formula: exp(-2 * ((t - T/2) / T)^2)
        Normalized so weights sum to num_steps.
        """
        x = self.timesteps

        # Gaussian centered at middle timestep
        y = torch.exp(-2 * ((x - num_steps / 2) / num_steps) ** 2)

        # Shift to make minimum 0
        y_shifted = y - y.min()

        # Normalize so weights sum to num_steps
        weights = y_shifted * (num_steps / y_shifted.sum())

        # Handle case where we have different number of timesteps than 1000
        if len(self.timesteps) != self.num_train_timesteps:
            weights = weights * (len(self.timesteps) / num_steps)
            # Add small offset to avoid zero weights
            weights = weights + weights[1] if len(weights) > 1 else weights + 0.1

        self.training_weights = weights

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to samples using flow matching interpolation.

        x_t = (1 - sigma) * x_0 + sigma * noise

        Args:
            original_samples: Clean samples x_0
            noise: Random noise epsilon
            timestep: Timestep values

        Returns:
            Noisy samples x_t
        """
        # Find closest timestep index
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)

        # Get sigma for each timestep
        timestep_ids = torch.argmin(
            (self.timesteps.unsqueeze(0).to(timestep.device) - timestep.unsqueeze(1)).abs(),
            dim=1
        )
        sigma = self.sigmas.to(timestep.device)[timestep_ids]

        # Reshape sigma for broadcasting
        while sigma.dim() < original_samples.dim():
            sigma = sigma.unsqueeze(-1)

        # Linear interpolation: x_t = (1 - sigma) * x_0 + sigma * noise
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the training target for flow matching.

        v_target = noise - x_0

        Args:
            sample: Clean samples x_0
            noise: Random noise epsilon
            timestep: Timestep values (unused, for API compatibility)

        Returns:
            Target velocity field
        """
        return noise - sample

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Get the training weight for a given timestep.

        Args:
            timestep: Timestep value

        Returns:
            Weight scalar for loss
        """
        timestep_id = torch.argmin(
            (self.timesteps.to(timestep.device) - timestep).abs()
        )
        return self.training_weights.to(timestep.device)[timestep_id]


class FlowMatchingLoss:
    """
    Flow Matching loss function matching DiffSynth-Studio implementation.

    Features:
    - Discrete timestep sampling via scheduler
    - Wan-specific sigma schedule with shift=5
    - Gaussian timestep-dependent loss weighting
    - Configurable timestep boundaries
    - TI2V support: fuse image latent into frame 0 with separated timesteps

    For TI2V (text-image-to-video):
    - Image latent is fused into frame 0 of the noisy video
    - Frame 0 gets timestep=0 (clean), rest get actual timestep
    - Loss is only computed on frames 1-N (frame 0 is preserved)

    Args:
        num_train_timesteps: Number of training timesteps
        shift: Timestep shift for Wan (default: 5.0)
        min_timestep_boundary: Minimum timestep as fraction of total (default: 0.0)
        max_timestep_boundary: Maximum timestep as fraction of total (default: 1.0)
        use_weighting: Whether to apply Gaussian timestep weighting (default: True)
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        min_timestep_boundary: float = 0.0,
        max_timestep_boundary: float = 1.0,
        use_weighting: bool = True,
    ):
        self.scheduler = FlowMatchScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
        )
        self.min_timestep_boundary = min_timestep_boundary
        self.max_timestep_boundary = max_timestep_boundary
        self.use_weighting = use_weighting

    def sample_timestep(self, device: torch.device) -> tuple:
        """
        Sample a random timestep within the configured boundaries.

        Args:
            device: Device for tensor

        Returns:
            Tuple of (timestep, timestep_index)
        """
        num_timesteps = len(self.scheduler.timesteps)

        # Compute boundary indices
        min_idx = int(self.min_timestep_boundary * num_timesteps)
        max_idx = int(self.max_timestep_boundary * num_timesteps)

        # Ensure valid range
        min_idx = max(0, min_idx)
        max_idx = min(num_timesteps, max_idx)
        if max_idx <= min_idx:
            max_idx = min_idx + 1

        # Sample random timestep index
        timestep_idx = torch.randint(min_idx, max_idx, (1,), device=device)
        timestep = self.scheduler.timesteps.to(device)[timestep_idx]

        return timestep, timestep_idx

    def __call__(
        self,
        model,
        x_0: list,
        context: list,
        seq_len: int,
        image_latents: list = None,
    ) -> torch.Tensor:
        """
        Compute the flow matching loss for TI2V training.

        For TI2V:
        - Image latent is fused into frame 0 of the noisy video
        - Frame 0 gets timestep=0 (preserved as conditioning)
        - Loss is computed on all frames but frame 0 has no noise

        Args:
            model: WanModel instance
            x_0: List of clean video latents, each [C, F, H, W]
            context: List of text embeddings, each [L, D]
            seq_len: Maximum sequence length for positional encoding
            image_latents: List of image latents for TI2V, each [C, 1, H, W]
                          If provided, fuses into frame 0 of noisy latent

        Returns:
            Scalar loss tensor
        """
        batch_size = len(x_0)
        device = x_0[0].device
        dtype = x_0[0].dtype

        # Move scheduler tensors to device
        self.scheduler.sigmas = self.scheduler.sigmas.to(device)
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)
        self.scheduler.training_weights = self.scheduler.training_weights.to(device)

        # Sample timesteps for each sample in batch
        timesteps = []
        timestep_indices = []
        for _ in range(batch_size):
            t, t_idx = self.sample_timestep(device)
            timesteps.append(t)
            timestep_indices.append(t_idx)

        # Stack timesteps: [B]
        t = torch.cat(timesteps)

        # Sample noise for each sample
        noise = [torch.randn_like(x) for x in x_0]

        # Compute noisy samples using scheduler
        x_t = []
        for i, (x, n) in enumerate(zip(x_0, noise)):
            # Get sigma for this timestep
            sigma = self.scheduler.sigmas[timestep_indices[i]]
            # x_t = (1 - sigma) * x_0 + sigma * noise
            x_noisy = (1 - sigma) * x + sigma * n

            # TI2V: Fuse image latent into frame 0 (replaces noisy frame 0)
            if image_latents is not None:
                # image_latents[i] shape: [C, 1, H, W]
                # Replace frame 0 with clean image latent
                x_noisy = x_noisy.clone()
                x_noisy[:, 0:1, :, :] = image_latents[i]

            x_t.append(x_noisy)

        # Compute target velocity: v = noise - x_0
        v_target = []
        for i, (x, n) in enumerate(zip(x_0, noise)):
            target = self.scheduler.training_target(x, n, None)

            # TI2V: Frame 0 target should be zero (no change needed for clean frame)
            if image_latents is not None:
                target = target.clone()
                target[:, 0:1, :, :] = 0.0

            v_target.append(target)

        # Model prediction
        # Convert timesteps to the format expected by WanModel (normalized to [0, 1])
        t_normalized = t / self.scheduler.num_train_timesteps
        v_pred = model(x_t, t_normalized, context, seq_len)

        # Compute weighted MSE loss
        total_loss = 0.0
        for i, (pred, target) in enumerate(zip(v_pred, v_target)):
            if image_latents is not None:
                # TI2V: Only compute loss on frames 1-N (skip frame 0)
                pred_frames = pred[:, 1:, :, :]
                target_frames = target[:, 1:, :, :]
                sample_loss = F.mse_loss(pred_frames.float(), target_frames.float())
            else:
                # T2V: Compute loss on all frames
                sample_loss = F.mse_loss(pred.float(), target.float())

            # Apply timestep-dependent weighting
            if self.use_weighting:
                weight = self.scheduler.training_weight(timesteps[i])
                sample_loss = sample_loss * weight

            total_loss = total_loss + sample_loss

        # Average over batch
        loss = total_loss / batch_size

        return loss
