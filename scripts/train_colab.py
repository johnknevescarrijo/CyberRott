#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Treinamento Fine-Tuning LoRA / SFT para o CyberRott 1.0
Baseado em microsoft/Phi-4-mini-instruct (3.8B) usando TRL, PEFT e Transformers.
Compatível com GPUs L4 (24GB), A100/H100 (40/80GB) no Google Colab Pro / Pro+.
"""

import os
import gc
import inspect
import argparse
import torch

# Otimizações de alocação de memória CUDA e desativação do W&B
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_DISABLED"] = "true"

# Limpeza prévia de VRAM
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

def parse_args():
    parser = argparse.ArgumentParser(
        description="CyberRott 1.0 - Treinamento LoRA / SFT com Phi-4-mini-instruct"
    )

    # Argumentos do Modelo
    model_group = parser.add_argument_group("Modelo")
    model_group.add_argument(
        "--model_name_or_path",
        type=str,
        default="microsoft/Phi-4-mini-instruct",
        help="Identificador do modelo base no Hugging Face Hub ou diretório local."
    )
    model_group.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Precisão do modelo base (padrão: bfloat16 para GPUs modernas)."
    )
    model_group.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Implementação de atenção: flash_attention_2 ou sdpa (Scaled Dot-Product nativo)."
    )

    # Argumentos dos Dados
    data_group = parser.add_argument_group("Dataset")
    data_group.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Nome do dataset no Hugging Face Hub (se aplicável)."
    )
    data_group.add_argument(
        "--data_file",
        type=str,
        default="cyberrott_final_train.jsonl",
        help="Caminho do arquivo JSONL local com os dados de treino."
    )
    data_group.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Janela máxima de contexto para treinamento (ex: 4096 ou 8192)."
    )

    # Argumentos LoRA
    lora_group = parser.add_argument_group("LoRA / PEFT")
    lora_group.add_argument(
        "--lora_r",
        type=int,
        default=32,
        help="Rank da matriz LoRA (padrão: 32)."
    )
    lora_group.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="Fator Alpha de escala LoRA (padrão: 64, normalmente 2x o rank)."
    )
    lora_group.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Taxa de Dropout para LoRA (padrão: 0.05)."
    )
    lora_group.add_argument(
        "--target_modules",
        type=str,
        default="all-linear",
        help=(
            "Módulos alvo para LoRA. Use 'all-linear' para cobrir qkv_proj, o_proj, "
            "gate_up_proj e down_proj do Phi-4, ou lista separada por vírgula."
        )
    )

    # Argumentos de Treinamento / SFTConfig
    train_group = parser.add_argument_group("Hiperparâmetros de Treinamento")
    train_group.add_argument(
        "--output_dir",
        type=str,
        default="./CyberRott-Phi4-LoRA",
        help="Diretório onde os pesos adaptadores LoRA serão salvos."
    )
    train_group.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Número total de épocas de treinamento."
    )
    train_group.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Batch size por GPU (2 para evitar OOM na cross_entropy com vocab 200k; 4 a 8 em 80GB)."
    )
    train_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Passos de acumulação de gradiente (batch efetivo = batch_size * grad_accum = 16)."
    )
    train_group.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Ativa gradient checkpointing para reduzir uso de VRAM de ativações (padrão: True)."
    )
    train_group.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="Ferramenta de rastreamento/métricas (padrão: none para evitar prompt do wandb)."
    )
    train_group.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Taxa de aprendizado (Learning Rate)."
    )
    train_group.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="Tipo de escalonador de LR (cosine, linear, etc.)."
    )
    train_group.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Fração de passos para warmup."
    )
    train_group.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Intervalo de passos para logging de perda (loss)."
    )
    train_group.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        choices=["epoch", "steps", "no"],
        help="Estratégia de salvamento de checkpoints."
    )
    train_group.add_argument(
        "--optim",
        type=str,
        default="adamw_torch_fused",
        help="Otimizador PyTorch (adamw_torch_fused, adamw_torch, etc.)."
    )

    return parser.parse_args()


def train():
    args = parse_args()

    print("=" * 70)
    print("Iniciando Pipeline de Treinamento - CyberRott 1.0")
    print(f"Modelo Base:           {args.model_name_or_path}")
    print(f"Hardware GPU:          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"DType selecionado:     {args.torch_dtype}")
    print(f"Target Modules LoRA:   {args.target_modules}")
    print(f"Diretório de Saída:    {args.output_dir}")
    print("=" * 70)

    # 1. Carregar Tokenizer
    # OBSERVAÇÃO CRÍTICA: trust_remote_code=False é explicitamente definido porque o
    # Phi-4-mini já possui suporte nativo no transformers. Usar True força o download de
    # arquivos obsoletos do hub que entram em conflito com versões recentes.
    print(f"1. Carregando Tokenizer para {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Carregar Modelo Base
    dtype = getattr(torch, args.torch_dtype)
    print(f"2. Carregando Modelo Base em {args.torch_dtype} com {args.attn_implementation}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
            device_map="auto",
            trust_remote_code=False
        )
    except Exception as e:
        print(f"[Aviso] Falha ao carregar com {args.attn_implementation}: {e}")
        print("Tentando fallback para 'sdpa' (Scaled Dot-Product Attention nativo)...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            device_map="auto",
            trust_remote_code=False
        )

    # 3. Configurar LoRA com target_modules adequados ao Phi-4
    # O Phi-4-mini usa arquitetura Phi3ForCausalLM com camadas fundidas:
    # Atenção: qkv_proj, o_proj | MLP: gate_up_proj, down_proj
    # 'all-linear' cobre automaticamente todas as projeções lineares de forma ótima.
    if args.target_modules.strip().lower() == "all-linear":
        target_modules = "all-linear"
    else:
        target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    print(f"3. LoRA configurado: r={args.lora_r}, alpha={args.lora_alpha}, target_modules={target_modules}")

    # 4. Carregar Dataset
    print("4. Carregando dataset...")
    if args.dataset_name:
        print(f"Carregando dataset do Hugging Face Hub: {args.dataset_name}")
        dataset = load_dataset(args.dataset_name, split="train")
    elif os.path.exists(args.data_file):
        print(f"Carregando dataset local: {args.data_file}")
        dataset = load_dataset("json", data_files=args.data_file, split="train")
    else:
        raise FileNotFoundError(
            f"Dataset não encontrado. Especifique --dataset_name ou garanta que '{args.data_file}' exista."
        )

    print(f"Dataset carregado com {len(dataset)} amostras.")

    # Formatar conversas para o template ChatML caso o dataset contenha 'messages'
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
        print("Formatando conversas usando o Chat Template do Phi-4...")
        dataset = dataset.map(format_chat, batched=True)

    # 5. Instanciar SFTConfig e SFTTrainer com verificação dinâmica de assinatura (TRL moderno)
    # TRL >= 0.15 renomeou 'max_seq_length' para 'max_length' em SFTConfig
    # e padronizou 'processing_class' no SFTTrainer.
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
    }

    # Tratamento max_seq_length vs max_length
    if "max_length" in sft_config_params:
        sft_kwargs["max_length"] = args.max_seq_length
    elif "max_seq_length" in sft_config_params:
        sft_kwargs["max_seq_length"] = args.max_seq_length

    # Tratamento dataset_text_field em SFTConfig
    if "dataset_text_field" in sft_config_params:
        sft_kwargs["dataset_text_field"] = "text"

    sft_config = SFTConfig(**sft_kwargs)

    # Preparar argumentos do SFTTrainer
    sft_trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "peft_config": peft_config,
        "args": sft_config,
    }

    # Compatibilidade processing_class vs tokenizer no SFTTrainer
    if "processing_class" in sft_trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sft_trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    # dataset_text_field direto no SFTTrainer caso a versão do TRL espere aqui
    if "dataset_text_field" in sft_trainer_params and "dataset_text_field" not in sft_kwargs:
        trainer_kwargs["dataset_text_field"] = "text"

    trainer = SFTTrainer(**trainer_kwargs)

    # 6. Executar Treinamento
    print("\n" + "=" * 70)
    print("Iniciando Treinamento LoRA / SFT...")
    print("=" * 70)
    trainer.train()

    # 7. Salvar Adaptadores LoRA e Tokenizer
    print(f"\nSalvando adaptadores LoRA e Tokenizer em: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("=" * 70)
    print("Treinamento finalizado com sucesso! Adaptadores prontos para inferência ou merge.")
    print("=" * 70)


if __name__ == "__main__":
    train()
