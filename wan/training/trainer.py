# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Training loop for Wan 2.2 finetuning with FSDP support.
"""

import os
import math
import logging
from typing import Optional
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

from .config import TrainingConfig
from .loss import FlowMatchingLoss

__all__ = ['Trainer']

logger = logging.getLogger(__name__)


def get_warmup_constant_schedule(
    optimizer,
    warmup_steps: int,
):
    """
    Create a learning rate scheduler with linear warmup then constant.

    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of warmup steps

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """
    Training loop for Wan 2.2 finetuning.

    Supports:
    - FSDP for multi-GPU/multi-node distributed training
    - Gradient accumulation
    - Mixed precision training (bf16/fp16)
    - Linear warmup + constant learning rate
    - On-the-fly VAE and T5 encoding

    Args:
        model: WanModel instance (already wrapped with FSDP if distributed)
        config: TrainingConfig instance
        vae: VAE encoder for video latent encoding (optional, for on-the-fly encoding)
        text_encoder: T5 encoder for text embedding (optional, for on-the-fly encoding)
    """

    def __init__(
        self,
        model,
        config: TrainingConfig,
        vae=None,
        text_encoder=None,
    ):
        self.model = model
        self.config = config
        self.vae = vae
        self.text_encoder = text_encoder

        # Distributed setup
        self.is_distributed = dist.is_initialized()
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.global_rank = dist.get_rank() if self.is_distributed else 0
        self.is_main_process = self.global_rank == 0

        # Device
        self.device = torch.device(f'cuda:{self.local_rank}')

        # Loss function with DiffSynth-style scheduler
        self.loss_fn = FlowMatchingLoss(
            num_train_timesteps=config.num_train_timesteps,
            shift=config.timestep_shift,
            min_timestep_boundary=config.min_timestep_boundary,
            max_timestep_boundary=config.max_timestep_boundary,
            use_weighting=config.use_timestep_weighting,
        )

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # Learning rate scheduler (warmup + constant)
        self.scheduler = get_warmup_constant_schedule(
            self.optimizer,
            warmup_steps=config.warmup_steps,
        )

        # Mixed precision
        self.use_amp = config.mixed_precision != 'no'
        if config.mixed_precision == 'bf16':
            self.amp_dtype = torch.bfloat16
        elif config.mixed_precision == 'fp16':
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.float32

        # Gradient scaler for fp16 (not needed for bf16)
        self.scaler = GradScaler() if config.mixed_precision == 'fp16' else None

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Compute sequence length for model forward pass
        # After VAE encoding: T -> (T-1)//4 + 1, H -> H//8, W -> W//8
        # After patch embedding: F_patches = T_latent, H_patches = H_latent//2, W_patches = W_latent//2
        t_latent = (config.num_frames - 1) // 4 + 1
        h_latent = config.height // 8
        w_latent = config.width // 8
        # Patch size is (1, 2, 2), so patches are:
        f_patches = t_latent
        h_patches = h_latent // 2
        w_patches = w_latent // 2
        self.seq_len = f_patches * h_patches * w_patches

        if self.is_main_process:
            logger.info(f"Computed seq_len: {self.seq_len} "
                       f"(F={f_patches}, H={h_patches}, W={w_patches})")

    def _encode_batch(self, batch: dict) -> tuple:
        """
        Encode a batch of videos, images, and prompts to latents and embeddings.

        For ti2v training, this encodes:
        - Videos -> video latents (target)
        - Images (first frame) -> image conditioning
        - Prompts -> text embeddings

        Args:
            batch: Dictionary with 'video', 'image', and 'prompt' keys

        Returns:
            Tuple of (video_latents, image_cond, text_embeddings) as lists
        """
        videos = batch['video']  # list of [C, T, H, W]
        images = batch['image']  # list of [C, H, W]
        prompts = batch['prompt']  # list of strings

        # Handle batched tensor vs list
        if isinstance(videos, torch.Tensor) and videos.dim() == 5:
            videos = [v for v in videos]
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            images = [img for img in images]

        # Move to device
        videos = [v.to(self.device) for v in videos]
        images = [img.to(self.device) for img in images]

        # VAE encoding
        with torch.no_grad():
            if self.vae is not None:
                # Encode videos
                video_latents = self.vae.encode(videos)

                # Encode images (add temporal dim for VAE: [C, H, W] -> [C, 1, H, W])
                images_4d = [img.unsqueeze(1) for img in images]
                image_latents = self.vae.encode(images_4d)

                # For TI2V: image_latents are used directly (shape [C, 1, H, W])
                # They will be fused into frame 0 of the noisy video in the loss function
                image_cond = image_latents  # Each is [C, 1, H, W]
            else:
                # If no VAE, assume inputs are already latents
                video_latents = videos
                # Create image latents from first frame
                image_cond = [img.unsqueeze(1) for img in images]  # [C, H, W] -> [C, 1, H, W]

        # Text encoding
        with torch.no_grad():
            if self.text_encoder is not None:
                text_embeddings = self.text_encoder(prompts, self.device)
            else:
                # If no text encoder, create dummy embeddings
                text_embeddings = [
                    torch.randn(256, 4096, device=self.device)
                    for _ in prompts
                ]

        return video_latents, image_cond, text_embeddings

    def train_step(self, batch: dict) -> float:
        """
        Execute a single training step for ti2v.

        Args:
            batch: Dictionary with 'video', 'image', and 'prompt' keys

        Returns:
            Loss value as float
        """
        # Encode batch (videos, images, prompts -> latents, conditioning, embeddings)
        video_latents, image_cond, text_embeddings = self._encode_batch(batch)

        # Mixed precision context
        amp_context = torch.amp.autocast('cuda', dtype=self.amp_dtype) if self.use_amp else nullcontext()

        # Forward pass with loss
        with amp_context:
            loss = self.loss_fn(
                model=self.model,
                x_0=video_latents,
                context=text_embeddings,
                seq_len=self.seq_len,
                y=image_cond,  # Image conditioning for ti2v
            )

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * self.config.gradient_accumulation_steps

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader yielding batches

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(dataloader):
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1

            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

                # Logging
                if self.is_main_process and self.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"Step {self.global_step} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()

                # Check max steps
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    break

        return total_loss / max(num_batches, 1)

    def train(self, dataset) -> None:
        """
        Main training loop.

        Args:
            dataset: Dataset yielding samples with 'video' and 'prompt' keys
        """
        # Create data loader
        if self.is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=True,
            )
        else:
            sampler = None

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

        # Training loop
        if self.is_main_process:
            print(f"\nStarting training:")
            print(f"  Epochs: {self.config.num_epochs}")
            print(f"  Batch size: {self.config.batch_size}")
            print(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
            print(f"  Effective batch size: {self.config.effective_batch_size * self.world_size}")
            print(f"  Learning rate: {self.config.learning_rate}")
            print(f"  Warmup steps: {self.config.warmup_steps}")
            print(f"  World size: {self.world_size}")
            print()

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch

            if self.is_distributed:
                sampler.set_epoch(epoch)

            if self.is_main_process:
                print(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            avg_loss = self.train_epoch(dataloader)

            if self.is_main_process:
                print(f"Epoch {epoch + 1} completed | Average Loss: {avg_loss:.4f}\n")

            # Check max steps
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

        # Final save
        self.save_checkpoint(final=True)

        if self.is_main_process:
            print("Training completed!")

    def _collate_fn(self, batch: list) -> dict:
        """
        Collate function for DataLoader.

        Keeps videos/images as lists (Wan model expects lists).
        """
        videos = [item['video'] for item in batch]
        images = [item['image'] for item in batch]
        prompts = [item['prompt'] for item in batch]
        return {'video': videos, 'image': images, 'prompt': prompts}

    def save_checkpoint(self, final: bool = False) -> None:
        """
        Save a training checkpoint including model weights.

        For FSDP models, gathers the full state dict on rank 0.

        Args:
            final: If True, save as 'final' checkpoint
        """
        os.makedirs(self.config.output_dir, exist_ok=True)

        if final:
            checkpoint_dir = os.path.join(self.config.output_dir, 'checkpoint_final')
        else:
            checkpoint_dir = os.path.join(
                self.config.output_dir,
                f'checkpoint_step_{self.global_step}'
            )

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Get model state dict - handle FSDP specially
        if isinstance(self.model, FSDP):
            # Use FSDP's full state dict - gathers on rank 0
            full_state_dict_config = FullStateDictConfig(
                offload_to_cpu=True,
                rank0_only=True,
            )
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                full_state_dict_config,
            ):
                model_state_dict = self.model.state_dict()

                # Only rank 0 saves
                if self.is_main_process:
                    model_path = os.path.join(checkpoint_dir, 'model.pt')
                    torch.save(model_state_dict, model_path)
        else:
            # Non-FSDP model - simple save
            if self.is_main_process:
                model_path = os.path.join(checkpoint_dir, 'model.pt')
                torch.save(self.model.state_dict(), model_path)

        # Save optimizer, scheduler, and training state (all ranks for FSDP optimizer)
        if self.is_main_process:
            training_state = {
                'global_step': self.global_step,
                'epoch': self.epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'config': self.config,
            }
            state_path = os.path.join(checkpoint_dir, 'training_state.pt')
            torch.save(training_state, state_path)
            print(f"Saved checkpoint to {checkpoint_dir}")

        # Synchronize all ranks before continuing
        if self.is_distributed:
            dist.barrier()

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Load a training checkpoint.

        Args:
            checkpoint_dir: Path to checkpoint directory
        """
        model_path = os.path.join(checkpoint_dir, 'model.pt')
        state_path = os.path.join(checkpoint_dir, 'training_state.pt')

        if not os.path.exists(model_path) or not os.path.exists(state_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_dir}")

        # Load training state
        training_state = torch.load(state_path, map_location='cpu')
        self.global_step = training_state['global_step']
        self.epoch = training_state['epoch']
        self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
        self.scheduler.load_state_dict(training_state['scheduler_state_dict'])

        # Load model state dict
        model_state_dict = torch.load(model_path, map_location='cpu')

        if isinstance(self.model, FSDP):
            # For FSDP, use full state dict loading
            full_state_dict_config = FullStateDictConfig(
                offload_to_cpu=True,
                rank0_only=True,
            )
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                full_state_dict_config,
            ):
                self.model.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)

        if self.is_main_process:
            print(f"Loaded checkpoint from {checkpoint_dir}")
            print(f"  Resuming from step {self.global_step}, epoch {self.epoch}")
