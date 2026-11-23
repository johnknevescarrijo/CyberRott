#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CyberRott 1.0 - LoRA / SFT Fine-Tuning Training Pipeline
========================================================
Based on microsoft/Phi-4-mini-instruct (3.82B) using TRL, PEFT, and Transformers.
Compatible with NVIDIA L4 (24GB) and A100/H100 (40/80GB) GPUs on Google Colab Pro / Pro+.
"""

import os
import re
import gc
import json
import shutil
import inspect
import argparse
from typing import Optional, Dict, Any, List, Tuple, Set
import torch

# CUDA memory allocation optimizations and W&B deactivation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_DISABLED"] = "true"

# Pre-emptive VRAM cleanup
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from peft import LoraConfig, PeftModel, TaskType
from trl import SFTTrainer, SFTConfig

try:
    from scripts.common import (
        SUPPORTED_BACKBONES,
        resolve_backbone_config,
        build_lora_config,
        setup_hardware_optimizations,
        DEFAULT_DRIVE_MOUNT,
        DEFAULT_DRIVE_CHECKPOINTS_DIR,
        ADAPTER_WEIGHT_FILES,
        HEAVY_TRAINING_STATE_PREFIXES,
        CHECKPOINT_ARTIFACT_HINT_FILES,
        _SCRATCH_RESUME_VALUES,
        has_complete_adapter,
        list_checkpoints,
        read_trainer_state,
        is_checkpoint_completed,
        _is_resumable_checkpoint_dir,
        _get_drive_subfolder_variants,
        resolve_drive_paths,
        promote_adapter_files,
        sync_checkpoint_to_google_drive,
        inspect_checkpoint_state,
        prepare_checkpoint_for_execution,
        _patch_trainer_state_batch_size,
        patch_checkpoint_batch_size,
        DriveSyncCallback,
    )
except ImportError:
    from common import (
        SUPPORTED_BACKBONES,
        resolve_backbone_config,
        build_lora_config,
        setup_hardware_optimizations,
        DEFAULT_DRIVE_MOUNT,
        DEFAULT_DRIVE_CHECKPOINTS_DIR,
        ADAPTER_WEIGHT_FILES,
        HEAVY_TRAINING_STATE_PREFIXES,
        CHECKPOINT_ARTIFACT_HINT_FILES,
        _SCRATCH_RESUME_VALUES,
        has_complete_adapter,
        list_checkpoints,
        read_trainer_state,
        is_checkpoint_completed,
        _is_resumable_checkpoint_dir,
        _get_drive_subfolder_variants,
        resolve_drive_paths,
        promote_adapter_files,
        sync_checkpoint_to_google_drive,
        inspect_checkpoint_state,
        prepare_checkpoint_for_execution,
        _patch_trainer_state_batch_size,
        patch_checkpoint_batch_size,
        DriveSyncCallback,
    )


def _load_tokenizer(model_or_adapter_path: str):
    """Load tokenizer with trust_remote_code=False and ensure pad_token is set."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_or_adapter_path,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_base_model(args: argparse.Namespace):
    """Load base causal LM with requested dtype/attention and safe fallback to sdpa."""
    dtype = getattr(torch, args.torch_dtype)
    try:
        return AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
            device_map="auto",
            trust_remote_code=False,
        )
    except Exception as e:
        print(f"[Warning] Failed to load with {args.attn_implementation}: {e}")
        print("Attempting fallback to 'sdpa' (native Scaled Dot-Product Attention)...")
        return AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            device_map="auto",
            trust_remote_code=False,
        )


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="CyberRott 1.0 - LoRA / SFT Training with Phi-4-mini-instruct"
    )

    # Model Arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=["phi4-mini", "qwen2.5-coder-7b", "phi4-14b"],
        help="Pre-configured backbone architecture from SUPPORTED_BACKBONES."
    )
    model_group.add_argument(
        "--model_name_or_path",
        type=str,
        default="microsoft/Phi-4-mini-instruct",
        help="Base model identifier on Hugging Face Hub or local path."
    )
    model_group.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Base model numerical precision (default: bfloat16 for modern GPUs)."
    )
    model_group.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation: flash_attention_2 or sdpa (native Scaled Dot-Product)."
    )

    # Data Arguments
    data_group = parser.add_argument_group("Dataset")
    data_group.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name on Hugging Face Hub (if applicable)."
    )
    data_group.add_argument(
        "--data_file", "--data_path",
        type=str,
        dest="data_file",
        default="cyberrott_final_train.jsonl",
        help="Local JSONL file path containing training data."
    )
    data_group.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence context window for training (default: 2048)."
    )

    # LoRA / PEFT Arguments
    lora_group = parser.add_argument_group("LoRA / PEFT")
    lora_group.add_argument(
        "--lora_r",
        type=int,
        default=None,
        help="LoRA rank dimension (default: 32)."
    )
    lora_group.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help="LoRA alpha scaling factor (default: 64, typically 2x rank)."
    )
    lora_group.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout rate (default: 0.05)."
    )
    lora_group.add_argument(
        "--target_modules",
        type=str,
        default="all-linear",
        help=(
            "Target modules for LoRA. Use 'all-linear' to cover qkv_proj, o_proj, "
            "gate_up_proj, and down_proj on Phi-4, or provide comma-separated list."
        )
    )
    lora_group.add_argument(
        "--use_dora",
        dest="use_dora",
        action="store_true",
        default=None,
        help="Enable Weight-Decomposed Low-Rank Adaptation (DoRA) for enhanced capacity."
    )
    lora_group.add_argument(
        "--no_dora",
        dest="use_dora",
        action="store_false",
        help="Explicitly disable DoRA even when --preset_v1_4 is active."
    )
    lora_group.add_argument(
        "--preset_v1_4", "--v1_4", "--v1.4",
        dest="preset_v1_4",
        action="store_true",
        default=False,
        help="Activate CyberRott v1.4 defaults (DoRA, LoRA r=64/alpha=128, max_seq_length=2048)."
    )

    # Training / SFTConfig Arguments
    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument(
        "--output_dir",
        type=str,
        default="./CyberRott-Phi4-LoRA",
        help="Directory where LoRA adapter weights will be saved."
    )
    train_group.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="auto",
        help="Resume strategy: 'auto' (default), 'none'/'false' to start fresh, or explicit checkpoint path."
    )
    train_group.add_argument(
        "--drive_dir",
        type=str,
        default=None,
        help="Optional Google Drive checkpoints directory (auto-detects /content/drive/MyDrive/CyberRott_Checkpoints if mounted)."
    )
    train_group.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints retained on disk (default: 2)."
    )
    train_group.add_argument(
        "--num_train_epochs", "--epochs",
        type=int,
        dest="num_train_epochs",
        default=3,
        help="Total number of training epochs (default: 3)."
    )
    train_group.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Per-GPU batch size (2 to prevent OOM in cross_entropy with 200k vocab; 4 to 8 on 80GB)."
    )
    train_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum = 16)."
    )
    train_group.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing to reduce activation VRAM (default: True)."
    )
    train_group.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="Tracking/metrics logging platform (default: none to avoid wandb prompts)."
    )
    train_group.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Optimizer learning rate (default: 2e-4)."
    )
    train_group.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="Learning rate scheduler type (cosine, linear, etc.)."
    )
    train_group.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Fraction of training steps dedicated to linear warmup."
    )
    train_group.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Step interval for loss logging."
    )
    train_group.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        choices=["epoch", "steps", "no"],
        help="Checkpoint save strategy."
    )
    train_group.add_argument(
        "--optim",
        type=str,
        default="adamw_torch_fused",
        help="PyTorch optimizer choice (adamw_torch_fused, adamw_torch, etc.)."
    )
    train_group.add_argument(
        "--auto_optimize",
        action="store_true",
        default=True,
        help="Auto-optimize batch size, TF32, and FlashAttention for A100/H100/L4 (default: True)."
    )
    train_group.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Inspect checkpoints and configuration without downloading weights or running training."
    )

    parsed = parser.parse_args(argv)
    if parsed.use_dora is None:
        parsed.use_dora = bool(parsed.preset_v1_4)
    if parsed.preset_v1_4:
        if parsed.lora_r is None:
            parsed.lora_r = 64
        if parsed.lora_alpha is None:
            parsed.lora_alpha = 128
    if parsed.lora_r is None:
        parsed.lora_r = 32
    if parsed.lora_alpha is None:
        parsed.lora_alpha = 64
    if parsed.backbone:
        cfg = resolve_backbone_config(parsed.backbone)
        if parsed.model_name_or_path == "microsoft/Phi-4-mini-instruct":
            parsed.model_name_or_path = cfg["model_id"]
        if parsed.target_modules == "all-linear":
            tm = cfg["target_modules"]
            parsed.target_modules = "all-linear" if tm == "all-linear" else ",".join(tm)
    return parsed


def train(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    if args is None:
        args = parse_args()
    hw_profile = setup_hardware_optimizations(args)

    print("=" * 70)
    print("Launching SFT Fine-Tuning Pipeline - CyberRott")
    print(f"Base Model:            {args.model_name_or_path}")
    print(f"GPU Hardware:          {hw_profile['device_name']} ({hw_profile['vram_gb']} GB VRAM)")
    print(f"Attention Kernel:      {args.attn_implementation} (TF32 Matmul: Active)")
    print(f"Effective Batch Size:  {args.per_device_train_batch_size} x {args.gradient_accumulation_steps} = {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"Selected DType:        {args.torch_dtype}")
    print(f"LoRA Target Modules:   {args.target_modules}")
    print(f"Output Directory:      {args.output_dir}")
    print(f"Resume Policy:         {getattr(args, 'resume_from_checkpoint', 'auto')}")
    print("=" * 70)

    # 0. Inspect existing local & Google Drive checkpoints before loading heavy models
    inspection = inspect_checkpoint_state(
        output_dir=args.output_dir,
        resume_from_checkpoint=getattr(args, "resume_from_checkpoint", "auto"),
        num_train_epochs=args.num_train_epochs,
        drive_dir=getattr(args, "drive_dir", None),
        drive_mount=DEFAULT_DRIVE_MOUNT,
    )
    prepared = prepare_checkpoint_for_execution(inspection, args.output_dir)

    if prepared["status"] == "completed":
        print(f"\n[✓] Training already completed! Final adapter ready at: '{args.output_dir}'. Skipping retraining.")
        if getattr(args, "dry_run", False):
            return {
                "status": "completed",
                "output_dir": args.output_dir,
                "resumed_from": None,
                "skipped_training": True,
                "dry_run": True,
            }
        has_local_tokenizer = any(
            os.path.isfile(os.path.join(args.output_dir, fname))
            for fname in ("tokenizer_config.json", "tokenizer.json", "tokenizer.model")
        )
        tokenizer_src = args.output_dir if has_local_tokenizer else args.model_name_or_path
        tokenizer = _load_tokenizer(tokenizer_src)
        base_model = _load_base_model(args)
        peft_model = PeftModel.from_pretrained(base_model, args.output_dir)
        print(f"[✓] Loaded and validated final SFT adapter from '{args.output_dir}'.")
        return {
            "status": "completed",
            "output_dir": args.output_dir,
            "resumed_from": None,
            "skipped_training": True,
            "model": peft_model,
            "tokenizer": tokenizer,
        }

    resume_checkpoint = prepared.get("resume_checkpoint")
    if getattr(args, "dry_run", False):
        print(f"[dry_run] Resolution status='{prepared['status']}', resume_checkpoint={resume_checkpoint}")
        return {
            "status": prepared["status"],
            "output_dir": args.output_dir,
            "resumed_from": resume_checkpoint,
            "skipped_training": True,
            "dry_run": True,
        }

    # 1. Load Tokenizer
    # CRITICAL NOTE: trust_remote_code=False is explicitly set because Phi-4-mini
    # has native support in transformers. Using True pulls legacy conflicting code.
    print(f"1. Loading Tokenizer for {args.model_name_or_path}...")
    tokenizer = _load_tokenizer(args.model_name_or_path)

    # 2. Load Base Model
    print(f"2. Loading Base Model in {args.torch_dtype} with {args.attn_implementation}...")
    model = _load_base_model(args)

    # 3. Configure LoRA targets tailored for base model
    if args.target_modules.strip().lower() == "all-linear":
        target_modules = "all-linear"
    else:
        target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]

    peft_config = build_lora_config(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        use_dora=getattr(args, "use_dora", False),
    )
    print(f"3. LoRA configured: r={args.lora_r}, alpha={args.lora_alpha}, target_modules={target_modules}, use_dora={getattr(args, 'use_dora', False)}")

    # 4. Load Dataset
    print("4. Loading dataset...")
    if args.dataset_name:
        print(f"Loading dataset from Hugging Face Hub: {args.dataset_name}")
        dataset = load_dataset(args.dataset_name, split="train")
    elif os.path.exists(args.data_file):
        print(f"Loading local dataset: {args.data_file}")
        dataset = load_dataset("json", data_files=args.data_file, split="train")
    else:
        raise FileNotFoundError(
            f"Dataset not found. Specify --dataset_name or ensure '{args.data_file}' exists."
        )

    print(f"Dataset loaded with {len(dataset)} samples.")

    # Format conversations to ChatML template if dataset contains 'messages'
    def format_chat(batch):
        if "messages" in batch:
            return {
                "text": [
                    tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
                    for msg in batch["messages"]
                ]
            }
        return batch

    if "messages" in dataset.column_names:
        print("Formatting conversations using Phi-4 Chat Template...")
        dataset = dataset.map(format_chat, batched=True)

    # 5. Instantiate SFTConfig and SFTTrainer with dynamic parameter inspection
    sft_config_params = inspect.signature(SFTConfig.__init__).parameters
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sft_config_params.values()
    )
    sft_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": getattr(args, "report_to", "none"),
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "bf16": (args.torch_dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "fp16": (args.torch_dtype == "float16"),
        "optim": args.optim,
        "dataloader_num_workers": 2 if torch.cuda.is_available() else 0,
        "dataloader_pin_memory": True if torch.cuda.is_available() else False,
        "group_by_length": True,
    }

    if "warmup_ratio" in sft_config_params or accepts_var_kwargs:
        sft_kwargs["warmup_ratio"] = args.warmup_ratio
    elif "warmup_steps" in sft_config_params and args.warmup_ratio:
        sft_kwargs["warmup_steps"] = args.warmup_ratio

    if ("save_total_limit" in sft_config_params or accepts_var_kwargs) and getattr(args, "save_total_limit", None) is not None:
        sft_kwargs["save_total_limit"] = args.save_total_limit

    if "max_length" in sft_config_params:
        sft_kwargs["max_length"] = args.max_seq_length
    elif "max_seq_length" in sft_config_params:
        sft_kwargs["max_seq_length"] = args.max_seq_length

    if "dataset_text_field" in sft_config_params:
        sft_kwargs["dataset_text_field"] = "text"

    if not accepts_var_kwargs:
        sft_kwargs = {k: v for k, v in sft_kwargs.items() if k in sft_config_params}

    sft_config = SFTConfig(**sft_kwargs)

    sft_trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "peft_config": peft_config,
        "args": sft_config,
    }

    if "processing_class" in sft_trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sft_trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    if "dataset_text_field" in sft_trainer_params and "dataset_text_field" not in sft_kwargs:
        trainer_kwargs["dataset_text_field"] = "text"

    trainer = SFTTrainer(**trainer_kwargs)

    drive_target_dir = prepared.get("drive_target_dir")
    if drive_target_dir and hasattr(trainer, "add_callback"):
        trainer.add_callback(
            DriveSyncCallback(
                drive_target_dir=drive_target_dir,
                save_total_limit=getattr(args, "save_total_limit", None),
            )
        )

    # 6. Execute Training
    print("\n" + "=" * 70)
    if resume_checkpoint:
        print(f"Resuming LoRA / SFT Training from: {resume_checkpoint}")
    else:
        print("Starting LoRA / SFT Training from scratch...")
    print("=" * 70)
    patch_checkpoint_batch_size(
        resume_checkpoint,
        args.per_device_train_batch_size,
        output_dir=args.output_dir,
        drive_dir=getattr(args, "drive_dir", None),
        drive_mount=DEFAULT_DRIVE_MOUNT,
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # 7. Save LoRA Adapters and Tokenizer
    print(f"\nSaving LoRA adapters and Tokenizer to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if drive_target_dir:
        sync_checkpoint_to_google_drive(
            args.output_dir,
            drive_dir=drive_target_dir,
            drive_mount=DEFAULT_DRIVE_MOUNT,
            include_checkpoints=False,
        )

    print("=" * 70)
    print("Training completed successfully! Adapters ready for inference or merging.")
    print("=" * 70)
    return {
        "status": "trained",
        "output_dir": args.output_dir,
        "resumed_from": resume_checkpoint,
        "skipped_training": False,
    }


if __name__ == "__main__":
    train()
