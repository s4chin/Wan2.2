#!/usr/bin/env python3
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Main training script for Wan 2.2 finetuning.

Usage:
    # Single GPU
    torchrun --nproc_per_node=1 train.py --model_path /path/to/model --use_dummy_data

    # Multi-GPU (single node)
    torchrun --nproc_per_node=8 train.py --model_path /path/to/model --use_dummy_data

    # Multi-node (example with 2 nodes)
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=<master> --master_port=29500 \
        train.py --model_path /path/to/model --use_dummy_data
"""

import argparse
import logging
import os
import random

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model

from wan.configs.wan_ti2v_5B import ti2v_5B as model_config
from wan.modules.model import WanModel
from wan.modules.vae2_2 import Wan2_2_VAE
from wan.modules.t5 import T5EncoderModel
from wan.distributed.fsdp import shard_model
from wan.data.dataset import DummyVideoDataset
from wan.training.config import TrainingConfig
from wan.training.trainer import Trainer

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Wan 2.2 Finetuning')

    # Model paths
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to pretrained Wan model directory')
    parser.add_argument('--vae_path', type=str, default=None,
                        help='Path to VAE checkpoint (default: <model_path>/Wan2.2_VAE.pth)')
    parser.add_argument('--t5_path', type=str, default=None,
                        help='Path to T5 checkpoint (default: <model_path>/models_t5_umt5-xxl-enc-bf16.pth)')

    # Data
    parser.add_argument('--use_dummy_data', action='store_true',
                        help='Use dummy dataset for testing')
    parser.add_argument('--data_root', type=str, default='',
                        help='Root directory for video data')
    parser.add_argument('--metadata_path', type=str, default='',
                        help='Path to metadata CSV/JSON')

    # Video config
    parser.add_argument('--num_frames', type=int, default=81,
                        help='Number of video frames')
    parser.add_argument('--height', type=int, default=480,
                        help='Video height')
    parser.add_argument('--width', type=int, default=832,
                        help='Video width')

    # Training
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Per-GPU batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Warmup steps')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Maximum training steps (overrides num_epochs)')

    # Memory
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing')
    parser.add_argument('--mixed_precision', type=str, default='bf16',
                        choices=['bf16', 'fp16', 'no'],
                        help='Mixed precision mode')

    # Distributed
    parser.add_argument('--use_fsdp', action='store_true', default=True,
                        help='Use FSDP for distributed training')

    # Output
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for checkpoints')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Save checkpoint every N steps')
    parser.add_argument('--logging_steps', type=int, default=10,
                        help='Log every N steps')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_dummy_samples', type=int, default=100,
                        help='Number of samples in dummy dataset')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint directory to resume from')

    # LoRA
    parser.add_argument('--lora_rank', type=int, default=0,
                        help='LoRA rank (0 = full finetune, >0 = LoRA)')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha (scaling factor)')
    parser.add_argument('--lora_dropout', type=float, default=0.0,
                        help='LoRA dropout')

    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training environment."""
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        return True
    return False


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    # Setup distributed
    is_distributed = setup_distributed()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    global_rank = dist.get_rank() if is_distributed else 0
    is_main = global_rank == 0

    if is_main:
        logger.info("Starting Wan 2.2 Finetuning")
        logger.info(f"Distributed: {is_distributed}")
        if is_distributed:
            logger.info(f"World size: {dist.get_world_size()}")

    # Set seed
    set_seed(args.seed)

    # Create config
    config = TrainingConfig(
        model_path=args.model_path,
        model_type='ti2v',
        vae_path=args.vae_path or os.path.join(args.model_path, 'Wan2.2_VAE.pth'),
        t5_path=args.t5_path or os.path.join(args.model_path, 'models_t5_umt5-xxl-enc-bf16.pth'),
        data_root=args.data_root,
        metadata_path=args.metadata_path,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        mixed_precision=args.mixed_precision,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_fsdp=args.use_fsdp,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        seed=args.seed,
    )

    device = torch.device(f'cuda:{local_rank}')

    # Load VAE
    if is_main:
        logger.info(f"Loading VAE from {config.vae_path}")
    vae = Wan2_2_VAE(
        vae_pth=config.vae_path,
        dtype=torch.float,
        device=device,
    )

    # Load T5 text encoder
    if is_main:
        logger.info(f"Loading T5 from {config.t5_path}")
    text_encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=config.t5_path,
        tokenizer_path='google/umt5-xxl',
    )

    # Load model
    if is_main:
        logger.info(f"Loading model from {args.model_path}")

    # Load the pretrained model
    model = WanModel.from_pretrained(args.model_path)
    model = model.to(device)

    if is_main:
        logger.info(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Apply LoRA if rank > 0
    if args.lora_rank > 0:
        if is_main:
            logger.info(f"Applying LoRA with rank={args.lora_rank}, alpha={args.lora_alpha}")

        # Target the attention layers (q, k, v, o projections)
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q", "k", "v", "o"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        if is_main:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"LoRA trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Enable gradient checkpointing if requested
    if config.gradient_checkpointing:
        if is_main:
            logger.info("Enabling gradient checkpointing")
        # Note: gradient checkpointing needs to be added to WanModel
        # For now, we'll enable it if the model supports it
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()

    # Wrap with FSDP if distributed
    if is_distributed and config.use_fsdp:
        if is_main:
            logger.info("Wrapping model with FSDP")
        model = shard_model(
            model,
            device_id=local_rank,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )

    # Create dataset
    if args.use_dummy_data:
        if is_main:
            logger.info("Using dummy dataset for testing")
        dataset = DummyVideoDataset(
            num_samples=args.num_dummy_samples,
            num_frames=config.num_frames,
            height=config.height,
            width=config.width,
        )
    else:
        raise NotImplementedError(
            "Real dataset not implemented yet. Use --use_dummy_data for testing."
        )

    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        vae=vae,
        text_encoder=text_encoder,
    )

    # Resume from checkpoint if specified
    if args.resume:
        if is_main:
            logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(dataset)

    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    main()
