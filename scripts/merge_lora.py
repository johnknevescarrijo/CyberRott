#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para mesclar (merge) os pesos do LoRA com o modelo base Phi-4-mini
e gerar um modelo standalone pronto para exportação (vLLM, GGUF/llama.cpp, HuggingFace Hub).
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_and_save(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    torch_dtype: str = "bfloat16",
    sft_adapter_path: str = None
):
    dtype = getattr(torch, torch_dtype)
    print("=" * 60)
    print("CyberRott - Fusão de Adaptadores LoRA (Merge & Unload)")
    print(f"Modelo Base:    {base_model_path}")
    if sft_adapter_path:
        print(f"Adaptador SFT:  {sft_adapter_path}")
    print(f"Adaptador RL:   {adapter_path}")
    print(f"Destino Final:  {output_path}")
    print(f"DType:          {torch_dtype}")
    print("=" * 60)

    print(f"1. Carregando modelo base: {base_model_path} em {torch_dtype}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=False
    )
    print("2. Carregando tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=False)
        print(f"   [✓] Tokenizer carregado a partir do adaptador: {adapter_path}")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=False)
        print(f"   [✓] Tokenizer carregado a partir do modelo base: {base_model_path}")

    # Se houver adaptador SFT intermediário, funde primeiro o SFT
    if sft_adapter_path and os.path.exists(sft_adapter_path) and os.path.exists(os.path.join(sft_adapter_path, "adapter_config.json")):
        print(f"3. Fundindo camada SFT intermediária: {sft_adapter_path}...")
        sft_model = PeftModel.from_pretrained(base_model, sft_adapter_path)
        base_model = sft_model.merge_and_unload()
        print("   [✓] Camada SFT fundida com sucesso no modelo base!")

    print(f"4. Fundindo adaptadores finais de: {adapter_path}...")
    final_lora_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = final_lora_model.merge_and_unload()
    print("   [✓] Camada final fundida com sucesso!")

    print(f"5. Salvando modelo completo standalone em: {output_path}...")
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    print("=" * 60)
    print("Sucesso! Modelo CyberRott Standalone salvo e pronto para exportação (GGUF / Ollama / vLLM).")
    print("=" * 60)


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser(description="Fusão LoRA para Phi-4-mini CyberRott")
    parser.add_argument("--base_model", default="microsoft/Phi-4-mini-instruct", help="Caminho do modelo base")
    parser.add_argument("--sft_adapter", default=None, help="Caminho do adaptador SFT prévio (opcional para modelos GRPO)")
    parser.add_argument("--adapter", required=True, help="Pasta contendo os adaptadores LoRA treinados (GRPO ou SFT)")
    parser.add_argument("--output", default="./CyberRott-v1.2-Standalone", help="Pasta de destino para o modelo fundido")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Precisão dos pesos fundidos")
    args = parser.parse_args()

    merge_and_save(args.base_model, args.adapter, args.output, args.torch_dtype, args.sft_adapter)
