#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CyberRott 1.0 - LoRA / SFT Fine-Tuning Training Pipeline
========================================================
Based on microsoft/Phi-4-mini-instruct (3.82B) using TRL, PEFT, and Transformers.
Compatible with NVIDIA L4 (24GB) and A100/H100 (40/80GB) GPUs on Google Colab Pro / Pro+.
"""

import os
import gc
import inspect
import argparse
from typing import Optional, Dict, Any
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
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

# ==============================================================================
# Backbone Scaling Catalog (v1.4)
# ==============================================================================

SUPPORTED_BACKBONES: Dict[str, Dict[str, Any]] = {
    "phi4-mini": {
        "model_id": "microsoft/Phi-4-mini-instruct",
        "target_modules": "all-linear",
        "stop_tokens": ["<|end|>", "<|endoftext|>"],
        "default_max_seq_length": 4096,
        "description": "Microsoft Phi-4-mini (3.82B) dense reasoning architecture",
    },
    "qwen2.5-coder-7b": {
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
        "default_max_seq_length": 4096,
        "description": "Qwen 2.5 Coder 7B Instruct specialized in code audit & reversing",
    },
    "phi4-14b": {
        "model_id": "microsoft/Phi-4",
        "target_modules": "all-linear",
        "stop_tokens": ["<|im_end|>", "<|end|>", "<|endoftext|>"],
        "default_max_seq_length": 4096,
        "description": "Microsoft Phi-4 14B deep deductive reasoning model",
    },
}


def resolve_backbone_config(backbone_or_model: str) -> Dict[str, Any]:
    """Resolve a chave de backbone ou ID de modelo Hugging Face para a configuração canônica."""
    if not backbone_or_model:
        return SUPPORTED_BACKBONES["phi4-mini"]
    key = backbone_or_model.lower().strip()
    if key in SUPPORTED_BACKBONES:
        return SUPPORTED_BACKBONES[key]
    for b_key, b_cfg in SUPPORTED_BACKBONES.items():
        if b_cfg["model_id"].lower() == key:
            return b_cfg
    return {
        "model_id": backbone_or_model,
        "target_modules": "all-linear",
        "stop_tokens": ["<|end|>", "<|endoftext|>", "<|im_end|>"],
        "default_max_seq_length": 4096,
        "description": f"Custom architecture: {backbone_or_model}",
    }


def build_lora_config(
    r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: Any = "all-linear",
    use_dora: bool = False,
    bias: str = "none",
    task_type: Any = TaskType.CAUSAL_LM,
) -> LoraConfig:
    """Instancia LoraConfig com suporte transparente a DoRA e target modules flexíveis."""
    lora_kwargs = {
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "bias": bias,
        "task_type": task_type,
    }
    if use_dora:
        sig = inspect.signature(LoraConfig.__init__).parameters
        if "use_dora" in sig:
            lora_kwargs["use_dora"] = True
    return LoraConfig(**lora_kwargs)


def setup_hardware_optimizations(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    """
    Detects and applies modern hardware accelerations for modern GPUs (A100, H100, L4):
    - TF32 across cuDNN and matmul (up to 3x tensor throughput without precision loss)
    - Preserves micro-batch at 2 and grad_accum at 8 (due to Phi-4's 200k vocabulary)
    - Automatic detection and preference for FlashAttention-2
    """
    profile: Dict[str, Any] = {
        "device_name": "CPU",
        "vram_gb": 0.0,
        "is_ampere_or_newer": False,
        "is_high_vram": False,
        "recommended_attn": "sdpa",
    }
    if not torch.cuda.is_available():
        return profile

    # Enable TF32 for Ampere / Ada / Hopper architectures (A100, H100, L4, RTX 30/40)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    major, _ = torch.cuda.get_device_capability(0)

    profile["device_name"] = device_name
    profile["vram_gb"] = round(total_vram_gb, 1)
    profile["is_ampere_or_newer"] = (major >= 8)
    profile["is_high_vram"] = (total_vram_gb >= 35.0)  # A100 40GB/80GB, H100 80GB

    try:
        import flash_attn  # noqa: F401
        profile["recommended_attn"] = "flash_attention_2"
    except ImportError:
        profile["recommended_attn"] = "sdpa"

    if args is not None and getattr(args, "auto_optimize", True):
        if profile["is_high_vram"]:
            # NOTE: Micro-batch MUST remain 2 (with gradient_accumulation_steps=8) even on A100 40GB,
            # because Phi-4's 200,064-token vocabulary generates FP32 logit tensors of size
            # [batch * seq_len * 200064 * 4 bytes]. batch=4 with long seq_len induces ~12.2 GiB OOM
            # in ForCausalLMLoss.
            if args.attn_implementation == "sdpa" and profile["recommended_attn"] == "flash_attention_2":
                args.attn_implementation = "flash_attention_2"
                print("[A100 Auto-Optimization] Automatically activated 'flash_attention_2'.")

    return profile


def parse_args():
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
        default=32,
        help="LoRA rank dimension (default: 32)."
    )
    lora_group.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
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
        action="store_true",
        default=False,
        help="Enable Weight-Decomposed Low-Rank Adaptation (DoRA) for enhanced capacity."
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

    parsed = parser.parse_args()
    if parsed.preset_v1_4:
        parsed.use_dora = True
        parsed.lora_r = 64
        parsed.lora_alpha = 128
    if parsed.backbone:
        cfg = resolve_backbone_config(parsed.backbone)
        if parsed.model_name_or_path == "microsoft/Phi-4-mini-instruct":
            parsed.model_name_or_path = cfg["model_id"]
        if parsed.target_modules == "all-linear":
            tm = cfg["target_modules"]
            parsed.target_modules = "all-linear" if tm == "all-linear" else ",".join(tm)
    return parsed


def train():
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
    print("=" * 70)

    # 1. Load Tokenizer
    # CRITICAL NOTE: trust_remote_code=False is explicitly set because Phi-4-mini
    # has native support in transformers. Using True pulls legacy conflicting code.
    print(f"1. Loading Tokenizer for {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Base Model
    dtype = getattr(torch, args.torch_dtype)
    print(f"2. Loading Base Model in {args.torch_dtype} with {args.attn_implementation}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
            device_map="auto",
            trust_remote_code=False
        )
    except Exception as e:
        print(f"[Warning] Failed to load with {args.attn_implementation}: {e}")
        print("Attempting fallback to 'sdpa' (native Scaled Dot-Product Attention)...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            device_map="auto",
            trust_remote_code=False
        )

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
    sft_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": "none",
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "bf16": (args.torch_dtype == "bfloat16" and torch.cuda.is_bf16_supported()),
        "fp16": (args.torch_dtype == "float16"),
        "optim": args.optim,
        "dataloader_num_workers": 2 if torch.cuda.is_available() else 0,
        "dataloader_pin_memory": True if torch.cuda.is_available() else False,
        "group_by_length": True,
    }

    if "max_length" in sft_config_params:
        sft_kwargs["max_length"] = args.max_seq_length
    elif "max_seq_length" in sft_config_params:
        sft_kwargs["max_seq_length"] = args.max_seq_length

    if "dataset_text_field" in sft_config_params:
        sft_kwargs["dataset_text_field"] = "text"

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

    # 6. Execute Training
    print("\n" + "=" * 70)
    print("Starting LoRA / SFT Training...")
    print("=" * 70)
    trainer.train()

    # 7. Save LoRA Adapters and Tokenizer
    print(f"\nSaving LoRA adapters and Tokenizer to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("=" * 70)
    print("Training completed successfully! Adapters ready for inference or merging.")
    print("=" * 70)


if __name__ == "__main__":
    train()
