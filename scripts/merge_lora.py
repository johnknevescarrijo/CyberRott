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

def merge_and_save(base_model_path: str, adapter_path: str, output_path: str, torch_dtype: str = "bfloat16"):
    dtype = getattr(torch, torch_dtype)
    print("=" * 60)
    print("CyberRott 1.0 - Fusão de Adaptadores LoRA (Merge & Unload)")
    print(f"Modelo Base:   {base_model_path}")
    print(f"Adaptador LoRA: {adapter_path}")
    print(f"Destino Final:  {output_path}")
    print(f"DType:          {torch_dtype}")
    print("=" * 60)

    print(f"1. Carregando modelo base: {base_model_path} em {torch_dtype}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="cpu",  # CPU para merge limpo sem estourar VRAM
        trust_remote_code=False
    )
    print("1. Carregando tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=False)
        print(f"   [✓] Tokenizer carregado a partir do adaptador: {adapter_path}")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=False)
        print(f"   [✓] Tokenizer carregado a partir do modelo base: {base_model_path}")

    print(f"2. Carregando adaptadores LoRA de: {adapter_path}...")
    lora_model = PeftModel.from_pretrained(base_model, adapter_path)

    print("3. Fundindo pesos (merge_and_unload)...")
    merged_model = lora_model.merge_and_unload()

    print(f"4. Salvando modelo completo standalone em: {output_path}...")
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    print("=" * 60)
    print("Sucesso! Modelo CyberRott Standalone salvo e pronto para deploy.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusão LoRA para Phi-4-mini CyberRott")
    parser.add_argument("--base_model", default="microsoft/Phi-4-mini-instruct", help="Caminho do modelo base")
    parser.add_argument("--adapter", required=True, help="Pasta contendo os adaptadores LoRA treinados")
    parser.add_argument("--output", default="./CyberRott-Phi4-mini-Standalone", help="Pasta de destino para o modelo fundido")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Precisão dos pesos fundidos")
    args = parser.parse_args()

    merge_and_save(args.base_model, args.adapter, args.output, args.torch_dtype)
