# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Training configuration for Wan 2.2 finetuning.
"""

from dataclasses import dataclass, field
from typing import Optional

__all__ = ['TrainingConfig']


@dataclass
class TrainingConfig:
    """
    Configuration for Wan 2.2 finetuning.

    Model Configuration:
        model_path: Path to pretrained Wan model checkpoint directory
        model_type: Model variant ('t2v', 'i2v', 'ti2v')
        vae_path: Path to VAE checkpoint
        t5_path: Path to T5 encoder checkpoint
        t5_tokenizer_path: Path to T5 tokenizer

    Data Configuration:
        data_root: Root directory containing video files
        metadata_path: Path to metadata CSV/JSON with video paths and captions
        num_frames: Number of video frames to use
        height: Video height in pixels
        width: Video width in pixels

    Training Configuration:
        batch_size: Per-GPU batch size
        gradient_accumulation_steps: Number of steps to accumulate gradients
        learning_rate: Peak learning rate
        warmup_steps: Number of warmup steps (linear warmup)
        num_epochs: Total training epochs
        max_steps: Maximum training steps (overrides num_epochs if set)

    Memory Optimization:
        gradient_checkpointing: Enable gradient checkpointing to save memory
        mixed_precision: Mixed precision mode ('bf16', 'fp16', 'no')

    Distributed Training:
        use_fsdp: Enable Fully Sharded Data Parallel

    Checkpointing:
        output_dir: Directory to save checkpoints
        save_steps: Save checkpoint every N steps
        logging_steps: Log metrics every N steps
    """

    # Model
    model_path: str = ""
    model_type: str = "ti2v"
    vae_path: str = ""
    t5_path: str = ""
    t5_tokenizer_path: str = "google/umt5-xxl"

    # Data
    data_root: str = ""
    metadata_path: str = ""
    num_frames: int = 81
    height: int = 480
    width: int = 832
    text_len: int = 512

    # Training
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    num_epochs: int = 10
    max_steps: Optional[int] = None

    # Memory optimization
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"

    # Loss configuration (DiffSynth-style)
    num_train_timesteps: int = 1000
    timestep_shift: float = 5.0  # Wan-specific shift
    min_timestep_boundary: float = 0.0  # Minimum timestep as fraction
    max_timestep_boundary: float = 1.0  # Maximum timestep as fraction
    use_timestep_weighting: bool = True  # Gaussian weighting on middle timesteps

    # Distributed
    use_fsdp: bool = True

    # Checkpointing
    output_dir: str = "./output"
    save_steps: int = 500
    logging_steps: int = 10

    # Seed
    seed: int = 42

    def __post_init__(self):
        """Validate configuration."""
        assert self.model_type in ['t2v', 'i2v', 'ti2v'], \
            f"model_type must be one of ['t2v', 'i2v', 'ti2v'], got {self.model_type}"
        assert self.mixed_precision in ['bf16', 'fp16', 'no'], \
            f"mixed_precision must be one of ['bf16', 'fp16', 'no'], got {self.mixed_precision}"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.gradient_accumulation_steps > 0, "gradient_accumulation_steps must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"

    @property
    def effective_batch_size(self) -> int:
        """Compute effective batch size accounting for gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})
