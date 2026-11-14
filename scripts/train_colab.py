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


DEFAULT_DRIVE_MOUNT = "/content/drive/MyDrive"
DEFAULT_DRIVE_CHECKPOINTS_DIR = "/content/drive/MyDrive/CyberRott_Checkpoints"
ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")
HEAVY_TRAINING_STATE_PREFIXES = (
    "optimizer",
    "scheduler",
    "rng_state",
    "scaler",
    "zero_to_fp32",
    "global_step",
)
CHECKPOINT_ARTIFACT_HINT_FILES = (
    "trainer_state.json",
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "optimizer.pt",
    "pytorch_model.bin",
    "model.safetensors",
)
_SCRATCH_RESUME_VALUES = {"none", "false", "no", "scratch", ""}


def has_complete_adapter(directory: str) -> bool:
    """Return True if directory contains adapter_config.json and adapter weights."""
    if not directory or not os.path.isdir(directory):
        return False
    cfg_path = os.path.join(directory, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        return False
    return any(os.path.isfile(os.path.join(directory, fname)) for fname in ADAPTER_WEIGHT_FILES)


def list_checkpoints(directory: str) -> List[Tuple[int, str]]:
    """Return numerically sorted (step, path) pairs for checkpoint-<step> subdirs."""
    if not directory or not os.path.isdir(directory):
        return []
    checkpoints: List[Tuple[int, str]] = []
    for entry in os.listdir(directory):
        match = re.fullmatch(r"checkpoint-(\d+)", entry)
        if not match:
            continue
        full_path = os.path.join(directory, entry)
        if os.path.isdir(full_path):
            checkpoints.append((int(match.group(1)), full_path))
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints


def read_trainer_state(checkpoint_dir: str) -> Optional[Dict[str, Any]]:
    """Read trainer_state.json from checkpoint_dir, returning None if absent or invalid."""
    state_path = os.path.join(checkpoint_dir, "trainer_state.json")
    if not os.path.isfile(state_path):
        return None
    try:
        with open(state_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def is_checkpoint_completed(
    checkpoint_dir: str,
    num_train_epochs: float,
    max_steps: Optional[int] = None,
) -> bool:
    """Return True when checkpoint has complete adapter files and finished training state."""
    if not has_complete_adapter(checkpoint_dir):
        return False
    state = read_trainer_state(checkpoint_dir)
    if state is None:
        return False

    state_epoch = state.get("epoch")
    state_step = state.get("global_step")
    state_max_steps = state.get("max_steps")
    state_num_epochs = state.get("num_train_epochs")

    if max_steps is not None and max_steps > 0 and state_step is not None:
        if int(state_step) >= int(max_steps):
            return True

    if state_epoch is not None and num_train_epochs is not None and float(num_train_epochs) > 0:
        if float(state_epoch) >= float(num_train_epochs) - 1e-3:
            return True

    if state_max_steps is not None and int(state_max_steps) > 0 and state_step is not None:
        if int(state_step) >= int(state_max_steps):
            if (
                state_num_epochs is None
                or num_train_epochs is None
                or float(state_num_epochs) >= float(num_train_epochs) - 1e-3
            ):
                return True

    return False


def _is_resumable_checkpoint_dir(checkpoint_dir: str) -> bool:
    """Return True if checkpoint_dir contains at least one recognized HF checkpoint artifact."""
    return any(
        os.path.isfile(os.path.join(checkpoint_dir, fname))
        for fname in CHECKPOINT_ARTIFACT_HINT_FILES
    )


def _get_drive_subfolder_variants(output_dir: str) -> Tuple[str, List[str]]:
    """Build canonical subfolder name and '-' / '_' / '.' variants for Drive lookup."""
    base = os.path.basename(os.path.normpath(output_dir)) if output_dir else ""
    target_subfolder = base if base and base not in (".", "/") else "CyberRott-SFT-LoRA"
    raw_variants = [
        target_subfolder,
        target_subfolder.replace("-", "_"),
        target_subfolder.replace("-", "_").replace(".", "_"),
        target_subfolder.replace("_", "-"),
    ]
    variants: List[str] = []
    for name in raw_variants:
        if name and name not in variants:
            variants.append(name)
    return target_subfolder, variants


def resolve_drive_paths(
    output_dir: str,
    drive_dir: Optional[str] = None,
    drive_mount: str = DEFAULT_DRIVE_MOUNT,
) -> Dict[str, Any]:
    """Resolve Google Drive root, target mirror directory, and existing candidate dirs."""
    target_subfolder, variants = _get_drive_subfolder_variants(output_dir)
    drive_root: Optional[str] = None
    drive_target_dir: Optional[str] = None
    candidate_dirs: List[str] = []

    if drive_dir:
        norm_drive = os.path.normpath(drive_dir)
        drive_base = os.path.basename(norm_drive)
        looks_like_run_dir = (
            drive_base in variants
            or has_complete_adapter(norm_drive)
            or len(list_checkpoints(norm_drive)) > 0
        )
        if looks_like_run_dir:
            drive_target_dir = norm_drive
            drive_root = os.path.dirname(norm_drive) or None
            candidate_dirs.append(norm_drive)
            if drive_root and os.path.isdir(drive_root):
                for variant in variants:
                    candidate_dirs.append(os.path.join(drive_root, variant))
        else:
            drive_root = norm_drive
            drive_target_dir = os.path.join(drive_root, target_subfolder)
            for variant in variants:
                candidate_dirs.append(os.path.join(drive_root, variant))
    elif os.path.exists(drive_mount) or os.path.exists(DEFAULT_DRIVE_CHECKPOINTS_DIR):
        drive_root = (
            os.path.join(drive_mount, "CyberRott_Checkpoints")
            if os.path.exists(drive_mount)
            else DEFAULT_DRIVE_CHECKPOINTS_DIR
        )
        drive_target_dir = os.path.join(drive_root, target_subfolder)
        for variant in variants:
            candidate_dirs.append(os.path.join(drive_root, variant))

    existing_candidates: List[str] = []
    seen_realpaths = set()
    for path in candidate_dirs:
        if os.path.isdir(path):
            real = os.path.realpath(path)
            if real not in seen_realpaths:
                seen_realpaths.add(real)
                existing_candidates.append(path)

    return {
        "target_subfolder": target_subfolder,
        "variants": variants,
        "drive_root": drive_root,
        "drive_target_dir": drive_target_dir,
        "candidate_drive_dirs": existing_candidates,
    }


def promote_adapter_files(src_dir: str, dst_dir: str) -> List[str]:
    """Copy adapter/tokenizer/config files from src_dir to dst_dir, skipping heavy training state."""
    if not os.path.isdir(src_dir):
        return []
    os.makedirs(dst_dir, exist_ok=True)
    copied: List[str] = []
    for entry in sorted(os.listdir(src_dir)):
        src_path = os.path.join(src_dir, entry)
        if not os.path.isfile(src_path):
            continue
        lower_name = entry.lower()
        if any(lower_name.startswith(prefix) for prefix in HEAVY_TRAINING_STATE_PREFIXES):
            continue
        dst_path = os.path.join(dst_dir, entry)
        if os.path.abspath(src_path) != os.path.abspath(dst_path):
            shutil.copy2(src_path, dst_path)
            copied.append(entry)
    return copied


def sync_checkpoint_to_google_drive(
    source_dir: str,
    target_subfolder: Optional[str] = None,
    drive_mount: str = DEFAULT_DRIVE_MOUNT,
    drive_dir: Optional[str] = None,
    include_checkpoints: bool = False,
) -> Optional[str]:
    """Synchronize final SFT adapter (and optionally checkpoints) to Google Drive."""
    drive_info = resolve_drive_paths(source_dir, drive_dir=drive_dir, drive_mount=drive_mount)
    drive_target_dir = drive_info["drive_target_dir"]
    drive_root = drive_info["drive_root"]

    if not drive_target_dir:
        return None
    if target_subfolder and drive_root and not drive_dir:
        drive_target_dir = os.path.join(drive_root, target_subfolder)
    if not os.path.exists(source_dir):
        print(f"[!] Source directory '{source_dir}' not found for Google Drive sync.")
        return None

    print(f"\n[*] Google Drive detected. Syncing '{source_dir}' -> '{drive_target_dir}'...")
    try:
        os.makedirs(drive_target_dir, exist_ok=True)
        if include_checkpoints:
            shutil.copytree(source_dir, drive_target_dir, dirs_exist_ok=True)
        else:
            promote_adapter_files(source_dir, drive_target_dir)

        if drive_root:
            sub_name = os.path.basename(os.path.normpath(drive_target_dir))
            for alt_candidate in {
                sub_name.replace("-", "_"),
                sub_name.replace("-", "_").replace(".", "_"),
            }:
                alt_path = os.path.join(drive_root, alt_candidate)
                if alt_path != drive_target_dir and not os.path.exists(alt_path):
                    try:
                        os.symlink(drive_target_dir, alt_path)
                    except Exception:
                        pass
        print(f"[✓] Final adapter synced to Google Drive: '{drive_target_dir}'!")
        return drive_target_dir
    except Exception as exc:
        print(f"[!] Warning: Failed to sync to Google Drive ({drive_target_dir}): {exc}")
        return None


def inspect_checkpoint_state(
    output_dir: str,
    resume_from_checkpoint: Optional[str] = "auto",
    num_train_epochs: float = 3,
    max_steps: Optional[int] = None,
    drive_dir: Optional[str] = None,
    drive_mount: str = DEFAULT_DRIVE_MOUNT,
) -> Dict[str, Any]:
    """Inspect local and Google Drive directories to determine resume/skip/scratch action."""
    drive_info = resolve_drive_paths(output_dir, drive_dir=drive_dir, drive_mount=drive_mount)
    drive_target_dir = drive_info["drive_target_dir"]
    drive_root = drive_info["drive_root"]
    candidate_drive_dirs = drive_info["candidate_drive_dirs"]

    local_checkpoints = list_checkpoints(output_dir)
    drive_checkpoints: List[Tuple[int, str]] = []
    for d_dir in candidate_drive_dirs:
        drive_checkpoints.extend(list_checkpoints(d_dir))
    drive_checkpoints.sort(key=lambda item: item[0])

    base_result: Dict[str, Any] = {
        "status": "scratch",
        "resume_checkpoint": None,
        "source_checkpoint": None,
        "completed_source": None,
        "completed_in_checkpoint": False,
        "needs_sync_from_drive": False,
        "local_checkpoints": local_checkpoints,
        "drive_checkpoints": drive_checkpoints,
        "drive_target_dir": drive_target_dir,
        "drive_root": drive_root,
        "candidate_drive_dirs": candidate_drive_dirs,
    }

    cleaned_arg = "auto" if resume_from_checkpoint is None else str(resume_from_checkpoint).strip()
    lowered = cleaned_arg.lower()

    if lowered in _SCRATCH_RESUME_VALUES:
        print("[*] Checkpoint auto-resume disabled. Starting training from scratch.")
        return base_result

    if lowered != "auto":
        explicit_path = cleaned_arg
        if not os.path.exists(explicit_path):
            in_output = os.path.join(output_dir, explicit_path)
            if os.path.exists(in_output):
                explicit_path = in_output
            else:
                for d_dir in candidate_drive_dirs:
                    in_drive = os.path.join(d_dir, explicit_path)
                    if os.path.exists(in_drive):
                        explicit_path = in_drive
                        break
        if os.path.exists(explicit_path):
            from_drive = any(
                os.path.commonpath([os.path.abspath(explicit_path), os.path.abspath(d_dir)]) == os.path.abspath(d_dir)
                for d_dir in candidate_drive_dirs
            )
            local_dest = (
                os.path.join(output_dir, os.path.basename(os.path.normpath(explicit_path)))
                if from_drive
                else explicit_path
            )
            base_result.update(
                {
                    "status": "resume",
                    "resume_checkpoint": local_dest,
                    "source_checkpoint": explicit_path,
                    "needs_sync_from_drive": from_drive and os.path.abspath(explicit_path) != os.path.abspath(local_dest),
                }
            )
            print(f"[✓] Explicit checkpoint selected: '{explicit_path}'.")
            return base_result
        print(f"[!] Specified checkpoint not found: '{cleaned_arg}'. Starting training from scratch.")
        return base_result

    # 1. Check root completion locally or on Google Drive
    if has_complete_adapter(output_dir):
        base_result.update(
            {
                "status": "completed",
                "completed_source": output_dir,
                "completed_in_checkpoint": False,
                "needs_sync_from_drive": False,
            }
        )
        print(f"[✓] Completed SFT adapter detected at local root: '{output_dir}'.")
        return base_result

    for d_dir in candidate_drive_dirs:
        if has_complete_adapter(d_dir):
            base_result.update(
                {
                    "status": "completed",
                    "completed_source": d_dir,
                    "completed_in_checkpoint": False,
                    "needs_sync_from_drive": True,
                }
            )
            print(f"[✓] Completed SFT adapter detected in Google Drive: '{d_dir}'.")
            return base_result

    # 2. Check whether any checkpoint-<step> has already reached target epochs/steps
    tagged_checkpoints: List[Tuple[int, int, str, bool]] = []
    for step, path in local_checkpoints:
        tagged_checkpoints.append((step, 1, path, False))
    for step, path in drive_checkpoints:
        tagged_checkpoints.append((step, 0, path, True))

    if not tagged_checkpoints:
        print(f"[*] No existing checkpoints found in '{output_dir}' or Google Drive. Starting from scratch.")
        return base_result

    # Sort descending by (step, local_priority) so highest step (and local over Drive on tie) comes first
    tagged_checkpoints.sort(key=lambda item: (item[0], item[1]), reverse=True)

    for step, _, ckpt_path, is_from_drive in tagged_checkpoints:
        if is_checkpoint_completed(ckpt_path, num_train_epochs=num_train_epochs, max_steps=max_steps):
            base_result.update(
                {
                    "status": "completed",
                    "completed_source": ckpt_path,
                    "completed_in_checkpoint": True,
                    "needs_sync_from_drive": is_from_drive,
                }
            )
            print(
                f"[✓] Completed training state detected in '{ckpt_path}' (step {step}). "
                "Will promote adapter to root and skip retraining."
            )
            return base_result

    # 3. Otherwise resume from the highest-step partial checkpoint
    resumable = [item for item in tagged_checkpoints if _is_resumable_checkpoint_dir(item[2])]
    pool = resumable if resumable else tagged_checkpoints
    best_step, _, best_path, is_from_drive = pool[0]
    local_resume_path = (
        os.path.join(output_dir, os.path.basename(best_path))
        if is_from_drive
        else best_path
    )
    base_result.update(
        {
            "status": "resume",
            "resume_checkpoint": local_resume_path,
            "source_checkpoint": best_path,
            "needs_sync_from_drive": is_from_drive,
        }
    )
    origin_label = "Google Drive" if is_from_drive else "local output_dir"
    print(f"[✓] Partial checkpoint found in {origin_label}: '{best_path}' (step {best_step}). Will resume.")
    return base_result


def prepare_checkpoint_for_execution(inspection: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Perform required promotions or Drive-to-local sync based on inspection result."""
    prepared = dict(inspection)
    status = prepared.get("status")
    drive_target_dir = prepared.get("drive_target_dir")

    if status == "completed":
        completed_src = prepared.get("completed_source")
        if completed_src and os.path.abspath(completed_src) != os.path.abspath(output_dir):
            copied = promote_adapter_files(completed_src, output_dir)
            print(f"[✓] Promoted/synced {len(copied)} adapter/tokenizer files from '{completed_src}' -> '{output_dir}'.")
        if drive_target_dir and not has_complete_adapter(drive_target_dir):
            sync_checkpoint_to_google_drive(output_dir, drive_dir=drive_target_dir, include_checkpoints=False)
        prepared["final_adapter_dir"] = output_dir
        return prepared

    if status == "resume" and prepared.get("needs_sync_from_drive"):
        src_ckpt = prepared.get("source_checkpoint")
        dst_ckpt = prepared.get("resume_checkpoint")
        if src_ckpt and dst_ckpt and os.path.abspath(src_ckpt) != os.path.abspath(dst_ckpt):
            os.makedirs(output_dir, exist_ok=True)
            print(f"[*] Syncing checkpoint from Google Drive: '{src_ckpt}' -> '{dst_ckpt}'...")
            shutil.copytree(src_ckpt, dst_ckpt, dirs_exist_ok=True)
            print(f"[✓] Checkpoint synced locally to '{dst_ckpt}'.")

    return prepared


def _patch_trainer_state_batch_size(state_path: str, target_batch_size: int) -> bool:
    """Update train_batch_size in a single trainer_state.json file if it differs from target_batch_size."""
    if not os.path.isfile(state_path):
        return False
    try:
        with open(state_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        print(f"[!] [VRAM Safeguard] Warning: Failed to read '{state_path}': {exc}")
        return False

    if not isinstance(data, dict):
        return False

    old_bs = data.get("train_batch_size")
    if old_bs == target_batch_size:
        return False

    data["train_batch_size"] = target_batch_size
    try:
        with open(state_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    except Exception as exc:
        print(f"[!] [VRAM Safeguard] Warning: Failed to write '{state_path}': {exc}")
        return False

    print(
        f"[VRAM Safeguard] Atualizado train_batch_size ({old_bs} -> {target_batch_size}) "
        f"em '{state_path}' para impedir que o Trainer restaure micro-batch maior ao retomar checkpoint."
    )
    return True


def patch_checkpoint_batch_size(
    checkpoint_dir: Optional[str],
    target_batch_size: int,
    output_dir: Optional[str] = None,
    drive_dir: Optional[str] = None,
    drive_mount: Optional[str] = None,
) -> bool:
    """Patch train_batch_size in trainer_state.json for checkpoint_dir, output_dir checkpoints, and Drive mirrors."""
    if not checkpoint_dir or target_batch_size is None:
        return False
    target_bs = int(target_batch_size)
    if target_bs <= 0:
        return False

    candidate_ckpt_dirs: List[str] = []
    seen_realpaths: Set[str] = set()

    def _add_ckpt_dir(path: Optional[str]) -> None:
        if not path or not os.path.isdir(path):
            return
        real = os.path.realpath(path)
        if real not in seen_realpaths:
            seen_realpaths.add(real)
            candidate_ckpt_dirs.append(path)

    norm_ckpt = os.path.normpath(checkpoint_dir)
    base = os.path.basename(norm_ckpt)
    ckpt_basename: Optional[str] = None
    inferred_output_dir: Optional[str] = None
    if re.fullmatch(r"checkpoint-\d+", base):
        ckpt_basename = base
        inferred_output_dir = os.path.dirname(norm_ckpt) or None
    _add_ckpt_dir(checkpoint_dir)

    effective_output_dir = output_dir or inferred_output_dir
    if effective_output_dir and os.path.isdir(effective_output_dir):
        for _, ckpt_path in list_checkpoints(effective_output_dir):
            _add_ckpt_dir(ckpt_path)

    effective_drive_mount = DEFAULT_DRIVE_MOUNT if drive_mount is None else drive_mount
    if effective_output_dir:
        drive_info = resolve_drive_paths(
            effective_output_dir,
            drive_dir=drive_dir,
            drive_mount=effective_drive_mount,
        )
        for d_dir in drive_info.get("candidate_drive_dirs", []):
            if ckpt_basename:
                _add_ckpt_dir(os.path.join(d_dir, ckpt_basename))
            for _, d_ckpt_path in list_checkpoints(d_dir):
                _add_ckpt_dir(d_ckpt_path)

    if not candidate_ckpt_dirs:
        return False

    updated = False
    for ckpt_path in candidate_ckpt_dirs:
        state_path = os.path.join(ckpt_path, "trainer_state.json")
        if _patch_trainer_state_batch_size(state_path, target_bs):
            updated = True

    return updated


class DriveSyncCallback(TrainerCallback):
    """Lightweight TrainerCallback that syncs each saved checkpoint-* to Google Drive."""

    def __init__(
        self,
        drive_target_dir: str,
        save_total_limit: Optional[int] = None,
    ) -> None:
        self.drive_target_dir = drive_target_dir
        self.save_total_limit = save_total_limit

    def on_save(self, args, state, control, **kwargs):
        if not self.drive_target_dir:
            return
        if not getattr(state, "is_world_process_zero", True):
            return
        output_dir = getattr(args, "output_dir", None)
        if not output_dir or not os.path.isdir(output_dir):
            return

        step = getattr(state, "global_step", None)
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}") if step is not None else None
        if not ckpt_dir or not os.path.isdir(ckpt_dir):
            checkpoints = list_checkpoints(output_dir)
            if not checkpoints:
                return
            step, ckpt_dir = checkpoints[-1]

        dest_ckpt = os.path.join(self.drive_target_dir, os.path.basename(ckpt_dir))
        try:
            os.makedirs(self.drive_target_dir, exist_ok=True)
            shutil.copytree(ckpt_dir, dest_ckpt, dirs_exist_ok=True)
            print(f"[✓] [DriveSync] Checkpoint 'checkpoint-{step}' synced to Google Drive: '{dest_ckpt}'")
            if self.save_total_limit is not None and self.save_total_limit > 0:
                drive_ckpts = list_checkpoints(self.drive_target_dir)
                if len(drive_ckpts) > self.save_total_limit:
                    for _, old_path in drive_ckpts[:-self.save_total_limit]:
                        shutil.rmtree(old_path, ignore_errors=True)
        except Exception as exc:
            print(f"[!] [DriveSync] Warning: Failed to sync checkpoint-{step} to Google Drive: {exc}")


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
        "bf16": (args.torch_dtype == "bfloat16" and torch.cuda.is_bf16_supported()),
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
