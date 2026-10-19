#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Inferência e Validação do CyberRott 1.0
Testa respostas com raciocínio (<think>) antes e depois do fine-tuning.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def run_inference(
    base_model_path: str = "microsoft/Phi-4-mini-instruct",
    adapter_path: str = None,
    prompt: str = "Explique como funciona um ataque de DLL Sideloading e como detectá-lo.",
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
):
    print("=" * 60)
    print("CyberRott 1.0 - Inferência e Validação de Raciocínio")
    print(f"Modelo Base: {base_model_path}")
    print(f"Adaptador:   {adapter_path if adapter_path else 'Nenhum (Modelo Base Puro)'}")
    print("=" * 60)

    # 1. Carregar Tokenizer (trust_remote_code=False para suporte nativo)
    print(f"Carregando tokenizer de: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=False)

    # 2. Carregar Modelo Base
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Carregando modelo base: {base_model_path} em {dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=False
    )

    # 3. Carregar Adaptadores LoRA se fornecido
    if adapter_path and os.path.exists(adapter_path):
        print(f"Carregando adaptadores LoRA de CyberRott: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    elif adapter_path:
        print(f"[Aviso] Caminho do adaptador '{adapter_path}' não encontrado. Rodando modelo base puro.")

    # 4. Formatar entrada no estilo CyberRott ChatML
    messages = [
        {
            "role": "system",
            "content": "Você é o CyberRott, uma IA especialista em segurança ofensiva, defensiva e análise forense com raciocínio detalhado."
        },
        {"role": "user", "content": prompt}
    ]

    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)

    print("\n" + "=" * 60)
    print(f"Prompt do Usuário:\n{prompt}")
    print("=" * 60)
    print("Gerando resposta com cadeia de pensamento...\n")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(response)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferência do CyberRott 1.0")
    parser.add_argument("--base_model", default="microsoft/Phi-4-mini-instruct", help="Caminho do modelo base")
    parser.add_argument("--adapter", default=None, help="Caminho da pasta do LoRA treinado (ex: ./CyberRott-Phi4-LoRA)")
    parser.add_argument("--prompt", default="Como analisar um arquivo pcap suspeito usando tshark em busca de exfiltração DNS?", help="Pergunta para o modelo")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Limite máximo de novos tokens gerados")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperatura de amostragem")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    args = parser.parse_args()

    run_inference(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
