#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for CyberRott training, checkpointing, Google Drive sync,
hardware optimization, and safe GPU/vLLM process cleanup.
"""

import argparse
import gc
import inspect
import json
import os
import re
import shutil
import signal
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch

try:
    from transformers import TrainerCallback as _HFTrainerCallback
except Exception:  # pragma: no cover - fallback when transformers unavailable
    class _HFTrainerCallback:
        pass


# ==============================================================================
# Backbone Scaling Catalog & LoRA / Hardware Helpers
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
    """Resolve a backbone alias or Hugging Face model ID to its canonical configuration."""
    if not backbone_or_model:
        return SUPPORTED_BACKBONES["phi4-mini"]
    key = backbone_or_model.lower().strip()
    if key in SUPPORTED_BACKBONES:
        return SUPPORTED_BACKBONES[key]
    for b_cfg in SUPPORTED_BACKBONES.values():
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
    task_type: Any = None,
) -> Any:
    """Instantiate LoraConfig with transparent DoRA support and flexible target modules."""
    from peft import LoraConfig, TaskType

    resolved_task_type = task_type if task_type is not None else TaskType.CAUSAL_LM
    lora_kwargs: Dict[str, Any] = {
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "bias": bias,
        "task_type": resolved_task_type,
    }
    if use_dora:
        sig = inspect.signature(LoraConfig.__init__).parameters
        if "use_dora" in sig:
            lora_kwargs["use_dora"] = True
    return LoraConfig(**lora_kwargs)


def setup_hardware_optimizations(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    """Detect GPU accelerator and configure TF32 and optimal attention mechanisms."""
    profile: Dict[str, Any] = {
        "device_name": "CPU",
        "vram_gb": 0.0,
        "is_ampere_or_newer": False,
        "is_high_vram": False,
        "recommended_attn": "sdpa",
    }
    if not torch.cuda.is_available():
        return profile

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    major, _ = torch.cuda.get_device_capability(0)

    profile["device_name"] = device_name
    profile["vram_gb"] = round(total_vram_gb, 1)
    profile["is_ampere_or_newer"] = major >= 8
    profile["is_high_vram"] = total_vram_gb >= 35.0

    try:
        import flash_attn  # noqa: F401

        profile["recommended_attn"] = "flash_attention_2"
    except ImportError:
        profile["recommended_attn"] = "sdpa"

    if args is not None and getattr(args, "auto_optimize", True):
        if (
            profile["is_high_vram"]
            and getattr(args, "attn_implementation", None) == "sdpa"
            and profile["recommended_attn"] == "flash_attention_2"
        ):
            args.attn_implementation = "flash_attention_2"
            print("[A100 Auto-Optimization] Automatically activated 'flash_attention_2'.")

    return profile


# ==============================================================================
# Checkpoint & Google Drive Constants and Helpers
# ==============================================================================

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
    if not os.path.isfile(os.path.join(directory, "adapter_config.json")):
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

    if max_steps is not None and int(max_steps) > 0 and state_step is not None:
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


def _get_drive_subfolder_variants(
    output_dir: str,
    fallback_name: str = "CyberRott-SFT-LoRA",
) -> Tuple[str, List[str]]:
    """Build canonical subfolder name and '-' / '_' / '.' variants for Drive lookup."""
    base = os.path.basename(os.path.normpath(output_dir)) if output_dir else ""
    target_subfolder = base if base and base not in (".", "/") else fallback_name
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
    fallback_name: str = "CyberRott-SFT-LoRA",
) -> Dict[str, Any]:
    """Resolve Google Drive root, target mirror directory, and existing candidate dirs."""
    target_subfolder, variants = _get_drive_subfolder_variants(output_dir, fallback_name=fallback_name)
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
    elif os.path.exists(drive_mount):
        drive_root = os.path.join(drive_mount, "CyberRott_Checkpoints")
        drive_target_dir = os.path.join(drive_root, target_subfolder)
        for variant in variants:
            candidate_dirs.append(os.path.join(drive_root, variant))

    existing_candidates: List[str] = []
    seen_realpaths: Set[str] = set()
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
    fallback_name: str = "CyberRott-SFT-LoRA",
) -> Optional[str]:
    """Synchronize final adapter (and optionally checkpoints) to Google Drive."""
    drive_info = resolve_drive_paths(
        source_dir,
        drive_dir=drive_dir,
        drive_mount=drive_mount,
        fallback_name=fallback_name,
    )
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
    fallback_name: str = "CyberRott-SFT-LoRA",
    label: str = "SFT",
) -> Dict[str, Any]:
    """Inspect local and Google Drive directories to determine resume/skip/scratch action."""
    drive_info = resolve_drive_paths(
        output_dir,
        drive_dir=drive_dir,
        drive_mount=drive_mount,
        fallback_name=fallback_name,
    )
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
            for base_dir in [output_dir, *candidate_drive_dirs]:
                cand = os.path.join(base_dir, explicit_path)
                if os.path.exists(cand):
                    explicit_path = cand
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
        print(f"[✓] Completed {label} adapter detected at local root: '{output_dir}'.")
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
            print(f"[✓] Completed {label} adapter detected in Google Drive: '{d_dir}'.")
            return base_result

    # 2. Check whether any checkpoint-<step> has already reached target epochs/steps
    tagged_checkpoints: List[Tuple[int, int, str, bool]] = [
        (step, 1, path, False) for step, path in local_checkpoints
    ] + [
        (step, 0, path, True) for step, path in drive_checkpoints
    ]

    if not tagged_checkpoints:
        print(f"[*] No existing checkpoints found in '{output_dir}' or Google Drive. Starting from scratch.")
        return base_result

    tagged_checkpoints.sort(key=lambda item: (item[0], item[1]), reverse=True)
    effective_max_steps = max_steps if (max_steps is not None and int(max_steps) > 0) else None

    for step, _, ckpt_path, is_from_drive in tagged_checkpoints:
        if is_checkpoint_completed(ckpt_path, num_train_epochs=num_train_epochs, max_steps=effective_max_steps):
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


def prepare_checkpoint_for_execution(
    inspection: Dict[str, Any],
    output_dir: str,
    label: str = "SFT",
) -> Dict[str, Any]:
    """Perform required promotions or Drive-to-local sync based on inspection result."""
    prepared = dict(inspection)
    status = prepared.get("status")
    drive_target_dir = prepared.get("drive_target_dir")

    if status == "completed":
        completed_src = prepared.get("completed_source")
        if completed_src and os.path.abspath(completed_src) != os.path.abspath(output_dir):
            copied = promote_adapter_files(completed_src, output_dir)
            print(f"[✓] Promoted/synced {len(copied)} {label} adapter files from '{completed_src}' -> '{output_dir}'.")
        if drive_target_dir and not has_complete_adapter(drive_target_dir):
            sync_checkpoint_to_google_drive(output_dir, drive_dir=drive_target_dir, include_checkpoints=False)
        prepared["final_adapter_dir"] = output_dir
        return prepared

    if status == "resume" and prepared.get("needs_sync_from_drive"):
        src_ckpt = prepared.get("source_checkpoint")
        dst_ckpt = prepared.get("resume_checkpoint")
        if src_ckpt and dst_ckpt and os.path.abspath(src_ckpt) != os.path.abspath(dst_ckpt):
            os.makedirs(output_dir, exist_ok=True)
            print(f"[*] Syncing {label} checkpoint from Google Drive: '{src_ckpt}' -> '{dst_ckpt}'...")
            shutil.copytree(src_ckpt, dst_ckpt, dirs_exist_ok=True)
            print(f"[✓] {label} checkpoint synced locally to '{dst_ckpt}'.")

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


class DriveSyncCallback(_HFTrainerCallback):
    """Lightweight TrainerCallback that syncs each saved checkpoint-* to Google Drive."""

    def __init__(
        self,
        drive_target_dir: str,
        save_total_limit: Optional[int] = None,
        log_prefix: str = "[DriveSync]",
    ) -> None:
        self.drive_target_dir = drive_target_dir
        self.save_total_limit = save_total_limit
        self.log_prefix = log_prefix

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
            print(f"[✓] {self.log_prefix} Checkpoint 'checkpoint-{step}' synced to Google Drive: '{dest_ckpt}'")
            if self.save_total_limit is not None and self.save_total_limit > 0:
                drive_ckpts = list_checkpoints(self.drive_target_dir)
                if len(drive_ckpts) > self.save_total_limit:
                    for _, old_path in drive_ckpts[:-self.save_total_limit]:
                        shutil.rmtree(old_path, ignore_errors=True)
        except Exception as exc:
            print(f"[!] {self.log_prefix} Warning: Failed to sync checkpoint-{step} to Google Drive: {exc}")


# ==============================================================================
# Safe GPU / vLLM Process Cleanup
# ==============================================================================

VLLM_PROCESS_PATTERNS: Tuple[str, ...] = (
    "trl.scripts.vllm_serve",
    "VLLM::EngineCore",
    "vllm.entrypoints",
    "vllm",
)


def _get_protected_pids() -> Set[int]:
    """Return current PID, parent PID, and all ancestor PIDs from /proc so cleanup never kills self or ancestors."""
    protected: Set[int] = set()
    pid = os.getpid()
    ppid = os.getppid()
    for p in (pid, ppid):
        if p and p > 0:
            protected.add(p)

    for start in (pid, ppid):
        cursor = start
        while cursor and cursor > 1:
            next_parent = None
            try:
                with open(f"/proc/{cursor}/status", "r", encoding="utf-8") as fh:
                    for line in fh:
                        if line.startswith("PPid:"):
                            next_parent = int(line.split(":", 1)[1].strip())
                            break
            except (OSError, ValueError):
                try:
                    with open(f"/proc/{cursor}/stat", "r", encoding="utf-8") as fh:
                        content = fh.read()
                    rparen = content.rfind(")")
                    if rparen != -1:
                        fields = content[rparen + 2:].split()
                        if len(fields) >= 2:
                            next_parent = int(fields[1])
                except (OSError, ValueError):
                    break
            if next_parent is None or next_parent <= 0 or next_parent in protected:
                break
            protected.add(next_parent)
            cursor = next_parent
    return protected


def _is_vllm_process(
    pid: int,
    patterns: Sequence[str] = VLLM_PROCESS_PATTERNS,
    *,
    proc_root: str = "/proc",
    os_path_isdir: Callable[[str], bool] = os.path.isdir,
    subprocess_run: Callable[..., Any] = subprocess.run,
) -> bool:
    """Return True only when pid's cmdline/comm matches a known vLLM pattern."""
    if pid <= 0:
        return False
    lowered_patterns = tuple(pat.lower() for pat in patterns)

    if os_path_isdir(proc_root):
        text_parts: List[str] = []
        for fname in ("cmdline", "comm"):
            try:
                with open(os.path.join(proc_root, str(pid), fname), "rb") as fh:
                    raw = fh.read()
                if raw:
                    text_parts.append(raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore"))
            except OSError:
                continue
        combined = " ".join(text_parts).lower()
        if not combined:
            return False
        return any(pat in combined for pat in lowered_patterns)

    # Safe fallback when /proc is unavailable: query ps for the specific PID; never guess.
    try:
        ps_proc = subprocess_run(
            ["ps", "-p", str(pid), "-o", "args="],
            check=False,
            capture_output=True,
            text=True,
        )
        if getattr(ps_proc, "returncode", 1) == 0:
            args_text = (getattr(ps_proc, "stdout", "") or "").lower()
            return any(pat in args_text for pat in lowered_patterns)
    except Exception:
        pass
    return False


def cleanup_external_vllm_server(
    *,
    protected_pids_fn: Callable[[], Set[int]] = _get_protected_pids,
    subprocess_run: Callable[..., Any] = subprocess.run,
    os_kill: Callable[[int, int], None] = os.kill,
    os_path_isdir: Callable[[str], bool] = os.path.isdir,
    time_sleep: Callable[[float], None] = time.sleep,
    cuda_mod: Any = None,
    proc_root: str = "/proc",
) -> bool:
    """Terminate lingering vLLM processes with SIGKILL while preserving non-vLLM GPU apps and ancestors."""
    protected_pids = protected_pids_fn()
    vllm_scan_patterns = ("trl.scripts.vllm_serve", "VLLM::EngineCore", "vllm.entrypoints")
    vllm_gpu_patterns = VLLM_PROCESS_PATTERNS
    killed_pids: List[int] = []
    pkill_killed = False

    # 1. Inspect active GPU compute processes via nvidia-smi and SIGKILL only verified vLLM PIDs
    try:
        smi_proc = subprocess_run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
        )
        smi_out = getattr(smi_proc, "stdout", "") or ""
        if getattr(smi_proc, "returncode", 1) == 0 and smi_out.strip():
            for raw_line in smi_out.splitlines():
                token = raw_line.strip()
                if not token.isdigit():
                    continue
                gpu_pid = int(token)
                if gpu_pid <= 0 or gpu_pid in protected_pids:
                    continue
                if not _is_vllm_process(
                    gpu_pid,
                    vllm_gpu_patterns,
                    proc_root=proc_root,
                    os_path_isdir=os_path_isdir,
                    subprocess_run=subprocess_run,
                ):
                    continue
                try:
                    os_kill(gpu_pid, signal.SIGKILL)
                    if gpu_pid not in killed_pids:
                        killed_pids.append(gpu_pid)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
    except Exception:
        pass

    # 2. Scan /proc for vLLM patterns (cmdline + comm) while protecting self & ancestors
    protected_matches_pattern = False
    if os_path_isdir(proc_root):
        try:
            proc_entries = os.listdir(proc_root)
        except OSError:
            proc_entries = []
        for entry in proc_entries:
            if not entry.isdigit():
                continue
            proc_pid = int(entry)
            if not _is_vllm_process(
                proc_pid,
                vllm_scan_patterns,
                proc_root=proc_root,
                os_path_isdir=os_path_isdir,
                subprocess_run=subprocess_run,
            ):
                continue
            if proc_pid in protected_pids:
                protected_matches_pattern = True
                continue
            try:
                os_kill(proc_pid, signal.SIGKILL)
                if proc_pid not in killed_pids:
                    killed_pids.append(proc_pid)
            except (ProcessLookupError, PermissionError, OSError):
                pass

    # 3. Fallback pkill -9 -f only when no protected ancestor matches any vLLM pattern
    if not protected_matches_pattern:
        try:
            proc = subprocess_run(
                ["pkill", "-9", "-f", "|".join(vllm_scan_patterns)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if getattr(proc, "returncode", 1) == 0:
                pkill_killed = True
        except Exception:
            pass

    if killed_pids or pkill_killed:
        if killed_pids:
            print(f"[VRAM Safeguard] Processos órfãos/vLLM encerrados com SIGKILL: {killed_pids}")
        else:
            print("[VRAM Safeguard] Processo externo residual vLLM encerrado com SIGKILL antes do modo colocate.")
        time_sleep(1.5)
        gc.collect()
        cuda = cuda_mod if cuda_mod is not None else torch.cuda
        if cuda.is_available():
            cuda.empty_cache()
            try:
                free_bytes, total_bytes = cuda.mem_get_info()
                free_gb = free_bytes / (1024 ** 3)
                total_gb = total_bytes / (1024 ** 3)
                print(f"[VRAM Safeguard] VRAM livre após limpeza: {free_gb:.2f} GiB / {total_gb:.2f} GiB.")
            except Exception:
                pass
        return True
    return False
