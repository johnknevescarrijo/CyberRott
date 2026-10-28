#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CyberRott - LoRA Adapter Weight Merging Pipeline (Merge & Unload)
================================================================
Consolidates low-rank delta matrices directly into base model weights:
1. Loads base model in specified precision (default: bfloat16).
2. Merges intermediate SFT LoRA layers if present.
3. Merges final GRPO / RL LoRA layers.
4. Cleans PEFT metadata and saves pure standalone Transformers model.
"""

import os
import sys

# Disable broken TensorFlow in Colab Python 3.13 environments
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
sys.modules["tensorflow"] = None

import argparse
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Disable torchao availability checks for Google Colab compatibility
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
    import peft.tuners.lora.torchao
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
except Exception:
    pass

from peft import PeftModel
import json

# Backbone Scaling Catalog (v1.4)
SUPPORTED_BACKBONES = {
    "phi4-mini": {
        "model_id": "microsoft/Phi-4-mini-instruct",
        "description": "Microsoft Phi-4-mini (3.82B) dense reasoning architecture",
    },
    "qwen2.5-coder-7b": {
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "description": "Qwen 2.5 Coder 7B Instruct specialized in code audit & reversing",
    },
    "phi4-14b": {
        "model_id": "microsoft/Phi-4",
        "description": "Microsoft Phi-4 14B deep deductive reasoning model",
    },
}


def resolve_backbone_config(backbone_or_model: str):
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
        "description": f"Custom architecture: {backbone_or_model}",
    }


def resolve_adapter_path(path: str) -> str:
    """Resolves adapter directory by inspecting direct paths, subdirectories, and candidate locations."""
    if not path:
        return None
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        return path
    if os.path.exists(path) and os.path.isdir(path):
        for root, _, files in os.walk(path):
            if "adapter_config.json" in files:
                return root
    # Common default candidates in Google Colab / local workspaces
    candidates = [
        "/content/drive/MyDrive/CyberRott_Checkpoints/CyberRott-v1.3-GRPO",
        "/content/drive/MyDrive/CyberRott_Checkpoints/CyberRott_v1_3_GRPO",
        "./CyberRott-v1.3-GRPO",
        "/content/drive/MyDrive/CyberRott_Checkpoints/CyberRott-v1.2-GRPO",
        "/content/drive/MyDrive/CyberRott_Checkpoints/CyberRott_v1_2_GRPO",
        "./CyberRott-v1.2-GRPO",
        "/content/drive/MyDrive/CyberRott_Checkpoints/cyberrott_v1_1_sft_lora",
        "/content/drive/MyDrive/CyberRott_Checkpoints/CyberRott_v1_1_Final",
        "./cyberrott_v1_1_sft_lora",
        "./CyberRott-v1.1-Final",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "adapter_config.json")):
            return c
    return None


def merge_and_save(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    torch_dtype: str = "bfloat16",
    sft_adapter_path: str = None
):
    dtype = getattr(torch, torch_dtype)

    resolved_sft = resolve_adapter_path(sft_adapter_path)
    resolved_final = resolve_adapter_path(adapter_path) or adapter_path

    print("=" * 60)
    print("CyberRott - LoRA Adapter Fusion (Merge & Unload)")
    print(f"Base Model:    {base_model_path}")
    if resolved_sft:
        print(f"SFT Adapter:   {resolved_sft}")
    else:
        print("SFT Adapter:   Not detected (direct merge)")
    print(f"RL Adapter:    {resolved_final}")
    print(f"Final Target:  {output_path}")
    print(f"DType:         {torch_dtype}")
    print("=" * 60)

    print(f"1. Loading base model: {base_model_path} in {torch_dtype}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=False
    )
    print("2. Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(resolved_final, trust_remote_code=False)
        print(f"   [✓] Tokenizer loaded from adapter: {resolved_final}")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=False)
        print(f"   [✓] Tokenizer loaded from base model: {base_model_path}")

    # If intermediate SFT adapter exists, fuse SFT first
    if resolved_sft:
        print(f"3. Merging intermediate SFT layer: {resolved_sft}...")
        sft_model = PeftModel.from_pretrained(base_model, resolved_sft)
        base_model = sft_model.merge_and_unload()
        # Clean residual peft_config to avoid warnings when loading next adapter
        if hasattr(base_model, "peft_config"):
            delattr(base_model, "peft_config")
        print("   [✓] SFT layer successfully merged into base model!")

    print(f"4. Merging final adapters from: {resolved_final}...")
    cfg_path = os.path.join(resolved_final, "adapter_config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                adapter_cfg = json.load(f)
                if adapter_cfg.get("use_dora", False):
                    print("   [*] Weight-Decomposed Low-Rank Adaptation (DoRA) detected in adapter. Applying DoRA directional/magnitude merge.")
        except Exception:
            pass

    final_lora_model = PeftModel.from_pretrained(base_model, resolved_final)
    merged_model = final_lora_model.merge_and_unload()
    # Clean residual peft_config to guarantee clean Transformers base model export
    if hasattr(merged_model, "peft_config"):
        delattr(merged_model, "peft_config")
    print("   [✓] Final adapter layer successfully merged!")

    print(f"5. Saving complete standalone model to: {output_path}...")
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    # Explicit memory deallocation to release VRAM immediately
    del merged_model, final_lora_model, base_model
    if "sft_model" in locals():
        del sft_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print("=" * 60)
    print("Success! CyberRott Standalone model saved and ready for export (GGUF / Ollama / vLLM).")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Fusion Pipeline for Phi-4-mini CyberRott")
    parser.add_argument(
        "--backbone",
        choices=["phi4-mini", "qwen2.5-coder-7b", "phi4-14b"],
        default=None,
        help="Pre-configured backbone architecture from SUPPORTED_BACKBONES."
    )
    parser.add_argument("--base_model", default="microsoft/Phi-4-mini-instruct", help="Base model identifier or local path")
    parser.add_argument("--sft_adapter", default=None, help="Path to prior SFT adapter (optional for GRPO models)")
    parser.add_argument("--adapter", required=True, help="Directory containing trained LoRA adapters (GRPO or SFT)")
    parser.add_argument("--output", default="./CyberRott-v1.3-Standalone", help="Target directory for the merged standalone model")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Numerical precision for merged weights")
    args = parser.parse_args()

    if args.backbone:
        cfg = resolve_backbone_config(args.backbone)
        if args.base_model == "microsoft/Phi-4-mini-instruct":
            args.base_model = cfg["model_id"]

    merge_and_save(args.base_model, args.adapter, args.output, args.torch_dtype, args.sft_adapter)
