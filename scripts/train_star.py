#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline de Treinamento Iterativo STaR (Self-Taught Reasoner) para CyberRott 1.1
================================================================================
Implementa o ciclo fechado de raciocínio auto-evoluído via RLVR / Rejection Sampling:
1. Rollout de Inferência (N=4 a 8 candidatos exploratórios com raciocínio <think>)
2. Verificação Determinística e Rejection Sampling (STaRVerifier)
3. Curadoria do Corpus de Treino (Filtragem estrita, ranking de qualidade e Âncoras de Ouro)
4. Atualização Iterativa SFT / LoRA (TRL SFTTrainer + PEFT)
5. Auto-Evolução: O modelo aprimorado na iteração i gera os rollouts da iteração i+1.

Suporta backends de rollout de alto desempenho (vLLM, Transformers nativo) e modo
determinístico (--dry_run / mock) para validação completa ponta a ponta sem GPU.
"""

import os
import sys
import gc
import re
import json
import time
import inspect
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from scripts.star_verifier import STaRVerifier, VerificationResult, CHOICE_MAP
except ImportError:
    from star_verifier import STaRVerifier, VerificationResult, CHOICE_MAP

import torch

# Otimizações de alocação de memória CUDA e desativação do W&B
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_DISABLED"] = "true"


def setup_hardware_optimizations(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    """
    Detecta e aplica acelerações de hardware de última geração para GPUs modernas (A100, H100, L4):
    - Ativação de TF32 em cuDNN e matmul (acelera até 3x operações de tensores sem perda de precisão)
    - Auto-sintonia de batch size em GPUs de alta capacidade (A100 40GB/80GB)
    - Detecção e preferência automática por FlashAttention-2
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

    # Ativação de TF32 para arquiteturas Ampere / Ada / Hopper (A100, H100, L4, RTX 30/40)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    major, _ = torch.cuda.get_device_capability(0)

    profile["device_name"] = device_name
    profile["vram_gb"] = round(total_vram_gb, 1)
    profile["is_ampere_or_newer"] = (major >= 8)
    profile["is_high_vram"] = (total_vram_gb >= 35.0)  # A100 40GB/80GB, H100 80GB

    # Verificar suporte a FlashAttention 2
    try:
        import flash_attn  # noqa: F401
        profile["recommended_attn"] = "flash_attention_2"
    except ImportError:
        profile["recommended_attn"] = "sdpa"

    if args is not None and getattr(args, "auto_optimize", True):
        # Em A100/H100 com >= 35GB VRAM, otimizar throughput de tensores
        if profile["is_high_vram"]:
            if args.per_device_train_batch_size == 2 and args.gradient_accumulation_steps == 8:
                args.per_device_train_batch_size = 4
                args.gradient_accumulation_steps = 4
                print(f"[Auto-Otimização A100 ({profile['vram_gb']}GB)] Ajustado batch_size=4, grad_accum=4 (Batch efetivo=16) para saturar os Tensor Cores.")
            if args.attn_implementation == "sdpa" and profile["recommended_attn"] == "flash_attention_2":
                args.attn_implementation = "flash_attention_2"
                print("[Auto-Otimização A100] Ativado 'flash_attention_2' automaticamente.")

    return profile


# ==============================================================================
# Argumentos de Linha de Comando (CLI)
# ==============================================================================

def parse_args(args_list: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="CyberRott 1.1 - Pipeline de Treinamento Iterativo STaR / Rejection Sampling"
    )

    # 1. Configurações de Dados e Problemas
    data_group = parser.add_argument_group("Problemas e Datasets")
    data_group.add_argument(
        "--problem_file", "--problem-file",
        dest="problem_file",
        type=str,
        default="data/dataset_cyber_ready.jsonl",
        help="Arquivo JSONL contendo a suíte de problemas/prompts para os rollouts."
    )
    data_group.add_argument(
        "--max_problems",
        type=int,
        default=None,
        help="Limita a quantidade de problemas processados (ideal para testes rápidos)."
    )
    data_group.add_argument(
        "--anchor_file", "--anchor-file",
        dest="anchor_file",
        type=str,
        default="data/dataset_cyber_ready.jsonl",
        help="Arquivo JSONL de amostras âncora de ouro para retenção de conhecimento."
    )
    data_group.add_argument(
        "--anchor_upsample", "--anchor-upsample",
        dest="anchor_upsample",
        type=int,
        default=5,
        help="Fator de repetição/upsampling das amostras âncora por iteração (0 para desativar)."
    )

    # 2. Configurações do Ciclo STaR
    star_group = parser.add_argument_group("Ciclo STaR e Rejection Sampling")
    star_group.add_argument(
        "--star_iterations", "--iterations", "--star-iterations",
        dest="star_iterations",
        type=int,
        default=1,
        help="Número de iterações do ciclo STaR (Rollout -> Verifier -> SFT Update)."
    )
    star_group.add_argument(
        "--num_candidates", "--num-candidates",
        dest="num_candidates",
        type=int,
        default=4,
        help="Número N de trajetórias exploratórias geradas por problema (padrão: 4, sugerido: 4 a 8)."
    )
    star_group.add_argument(
        "--max_verified_per_problem",
        type=int,
        default=1,
        help="Máximo de trajetórias verificadas mantidas por problema (Top-K aprovadas, padrão: 1)."
    )
    star_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperatura de amostragem para exploração lógica na geração de rollouts."
    )
    star_group.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling para os rollouts."
    )
    star_group.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Máximo de tokens gerados por candidato exploratório."
    )
    star_group.add_argument(
        "--rollout_backend", "--backend", "--rollout-backend",
        dest="rollout_backend",
        type=str,
        default="hf",
        choices=["hf", "vllm", "mock"],
        help="Backend de inferência para os rollouts ('vllm' para batching de alta velocidade, 'hf' para nativo, 'mock' para simulação)."
    )
    star_group.add_argument(
        "--dry_run", "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Modo de simulação rápida determinística (executa todo o pipeline em CPU sem carregar pesos de 4B)."
    )

    # 3. Configurações do Verificador Determinístico (STaRVerifier)
    verifier_group = parser.add_argument_group("Verificador Determinístico")
    verifier_group.add_argument(
        "--min_think_chars",
        type=int,
        default=80,
        help="Mínimo de caracteres exigidos no bloco <think>...</think>."
    )
    verifier_group.add_argument(
        "--strict_code_syntax",
        action="store_true",
        default=True,
        help="Rejeitar sumariamente trajetórias contendo código com erro de sintaxe."
    )
    verifier_group.add_argument(
        "--strict_ground_truth",
        action="store_true",
        default=True,
        help="Rejeitar sumariamente trajetórias que errarem o gabarito ou CWE esperado."
    )

    # 4. Configurações do Modelo e LoRA
    model_group = parser.add_argument_group("Modelo Base e LoRA")
    model_group.add_argument(
        "--model_name_or_path", "--model_path", "--model-path", "--model-name-or-path",
        dest="model_name_or_path",
        type=str,
        default="microsoft/Phi-4-mini-instruct",
        help="Modelo base inicial ou checkpoint LoRA prévio para a primeira iteração STaR."
    )
    model_group.add_argument(
        "--initial_adapter", "--adapter_path", "--adapter-path",
        dest="initial_adapter",
        type=str,
        default=None,
        help="Caminho opcional para adaptador LoRA existente (ex: checkpoint do CyberRott 1.0)."
    )
    model_group.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Precisão do modelo (padrão: bfloat16)."
    )
    model_group.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Implementação de atenção."
    )
    model_group.add_argument(
        "--lora_r",
        type=int,
        default=32,
        help="Rank LoRA (padrão: 32)."
    )
    model_group.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="Alpha LoRA (padrão: 64)."
    )
    model_group.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Dropout LoRA (padrão: 0.05)."
    )
    model_group.add_argument(
        "--target_modules",
        type=str,
        default="all-linear",
        help="Módulos alvo para LoRA (padrão: 'all-linear')."
    )

    # 5. Hiperparâmetros de Treinamento SFT
    train_group = parser.add_argument_group("Hiperparâmetros de Treinamento")
    train_group.add_argument(
        "--output_dir", "--output-dir",
        dest="output_dir",
        type=str,
        default="./CyberRott-Phi4-STaR",
        help="Diretório de saída para checkpoints, datasets curados e telemetria."
    )
    train_group.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Número de épocas de fine-tuning SFT por iteração STaR."
    )
    train_group.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Batch size por dispositivo no treino SFT."
    )
    train_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Passos de acumulação de gradientes."
    )
    train_group.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Taxa de aprendizado para o fine-tuning."
    )
    train_group.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Tamanho máximo de sequência para o SFTTrainer."
    )
    train_group.add_argument(
        "--logging_steps",
        type=int,
        default=5,
        help="Frequência de logging de loss."
    )
    train_group.add_argument(
        "--optim",
        type=str,
        default="adamw_torch_fused",
        help="Otimizador PyTorch."
    )
    train_group.add_argument(
        "--auto_optimize",
        action="store_true",
        default=True,
        help="Auto-otimiza batch size, TF32 e FlashAttention para A100/H100/L4 (padrão: True)."
    )

    return parser.parse_args(args_list)


# ==============================================================================
# Ingestão de Problemas e Criação de Prompts
# ==============================================================================

DEFAULT_SYSTEM_PROMPT = (
    "Você é o CyberRott, uma IA especialista em segurança ofensiva, defensiva e análise "
    "de vulnerabilidades com raciocínio analítico profundo."
)

def load_problem_suite(problem_file: str, max_problems: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Carrega a suíte de problemas de cibersegurança a partir de um JSONL,
    suportando formato ChatML ('messages') e formato de problemas diretos
    ('prompt', 'question', 'expected_cwe', 'answer', etc.).
    """
    if not os.path.exists(problem_file):
        fallback = os.path.join("data", problem_file)
        if os.path.exists(fallback):
            print(f"[*] Fallback de suíte de problemas: '{problem_file}' -> '{fallback}'")
            problem_file = fallback
        else:
            raise FileNotFoundError(f"Arquivo de problemas '{problem_file}' não encontrado.")

    problems = []
    with open(problem_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue

            problem_item: Dict[str, Any] = {
                "system_prompt": DEFAULT_SYSTEM_PROMPT,
                "user_prompt": "",
                "reference_answer": None,
                "metadata": dict(data),
            }

            # Caso 1: Formato ChatML com lista de 'messages'
            if "messages" in data and isinstance(data["messages"], list):
                for msg in data["messages"]:
                    role = msg.get("role")
                    content = msg.get("content", "")
                    if role == "system":
                        problem_item["system_prompt"] = content
                    elif role == "user":
                        problem_item["user_prompt"] = content
                    elif role == "assistant":
                        problem_item["reference_answer"] = content

            # Caso 2: Formato de benchmark de múltipla escolha (WMDP / SecQA)
            elif "question" in data and "choices" in data:
                q = data["question"].strip()
                choices = data["choices"]
                choices_formatted = "\n".join([f"{CHOICE_MAP.get(i, str(i))}. {c}" for i, c in enumerate(choices)])
                user_content = (
                    f"{q}\n\n{choices_formatted}\n\n"
                    "Instruções:\n"
                    "1. Realize sua análise técnica e cadeia de raciocínio dentro da tag <think>...</think>.\n"
                    "2. Após fechar </think>, declare a alternativa correta no formato exato: 'Resposta Final: [LETRA]'."
                )
                problem_item["user_prompt"] = user_content
                if "answer" in data:
                    problem_item["metadata"]["expected_answer"] = CHOICE_MAP.get(data["answer"], str(data["answer"]))

            # Caso 3: Formato com 'prompt' direto
            elif "prompt" in data:
                problem_item["user_prompt"] = data["prompt"]
                if "ground_truth" in data:
                    problem_item["reference_answer"] = data["ground_truth"]

            if problem_item["user_prompt"]:
                problems.append(problem_item)
                if max_problems and len(problems) >= max_problems:
                    break

    return problems


def load_gold_anchors(anchor_file: str, upsample_factor: int = 1) -> List[Dict[str, Any]]:
    """
    Carrega amostras âncora de ouro no formato ChatML e aplica upsampling,
    garantindo retenção de conhecimento e prevenção de catastrophic forgetting.
    """
    if not anchor_file or upsample_factor <= 0:
        return []

    if not os.path.exists(anchor_file):
        fallback = os.path.join("data", anchor_file)
        if os.path.exists(fallback):
            print(f"[*] Fallback de âncora STaR: '{anchor_file}' -> '{fallback}'")
            anchor_file = fallback
        else:
            return []

    anchors = []
    with open(anchor_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "messages" in data:
                    anchors.append(data)
            except Exception:
                continue

    if not anchors:
        return []

    # Aplica upsampling por duplicação
    upsampled = []
    for _ in range(upsample_factor):
        upsampled.extend(anchors)

    return upsampled


# ==============================================================================
# Fase 1: Gerador de Rollouts Exploratórios (Inference Rollout)
# ==============================================================================

class RolloutGenerator:
    """
    Gerencia a geração de N trajetórias candidatas por problema.
    Suporta os backends 'vllm', 'hf' (Hugging Face Transformers + PEFT) e 'mock'.
    Otimizado para prevenção de OOM na L4 (24GB) com descarregamento dinâmico de VRAM.
    """

    def __init__(
        self,
        backend: str,
        model_name_or_path: str,
        adapter_path: Optional[str] = None,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 1024,
    ):
        self.backend = backend
        self.model_name_or_path = model_name_or_path
        self.adapter_path = adapter_path
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        self.tokenizer = None
        self.model = None
        self.vllm_engine = None

        self._hf_model = None
        self._hf_tokenizer = None
        self._vllm_engine = None

        if self.backend != "mock":
            self._init_backend()

    def _init_hf(self):
        """
        Inicializa o backend Hugging Face com suporte nativo a Checkpoints LoRA:
        Se model_name_or_path contiver adapter_config.json, carrega o modelo base
        em bfloat16, anexa o adaptador PeftModel e executa merge_and_unload()
        para inferência ultrarrápida sem overhead de adapters.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        dtype = getattr(torch, self.torch_dtype)
        adapter_cfg_file = os.path.join(self.model_name_or_path, "adapter_config.json")
        is_lora_checkpoint = os.path.isdir(self.model_name_or_path) and os.path.isfile(adapter_cfg_file)

        if is_lora_checkpoint:
            try:
                with open(adapter_cfg_file, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                base_model_name = cfg_data.get("base_model_name_or_path", "microsoft/Phi-4-mini-instruct")
            except Exception:
                base_model_name = "microsoft/Phi-4-mini-instruct"

            print(f"[HF] Checkpoint LoRA detectado em: {self.model_name_or_path}")
            print(f"[HF] Carregando modelo base '{base_model_name}' em {self.torch_dtype}...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=dtype,
                attn_implementation=self.attn_implementation,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                trust_remote_code=False,
            )
            print(f"[HF] Mesclando pesos LoRA de '{self.model_name_or_path}' via merge_and_unload()...")
            peft_m = PeftModel.from_pretrained(base_model, self.model_name_or_path)
            self.model = peft_m.merge_and_unload()
            tokenizer_source = (
                self.model_name_or_path
                if os.path.exists(os.path.join(self.model_name_or_path, "tokenizer_config.json"))
                else base_model_name
            )
        else:
            print(f"[HF] Carregando modelo base em {self.torch_dtype} de: {self.model_name_or_path}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=dtype,
                attn_implementation=self.attn_implementation,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                trust_remote_code=False,
            )
            tokenizer_source = self.model_name_or_path

            # Se houver adaptador LoRA salvo adicional
            if self.adapter_path and os.path.exists(self.adapter_path):
                print(f"[HF] Carregando e mesclando adaptador LoRA anterior de: {self.adapter_path}")
                peft_m = PeftModel.from_pretrained(self.model, self.adapter_path)
                self.model = peft_m.merge_and_unload()

        print(f"[HF] Carregando tokenizer de: {tokenizer_source}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._hf_model = self.model
        self._hf_tokenizer = self.tokenizer
        self.model.eval()

    def _init_backend(self):
        """Inicializa o backend selecionado (vLLM ou HF Transformers)."""
        if self.backend == "vllm":
            try:
                from vllm import LLM
                from transformers import AutoTokenizer

                print(f"[vLLM] Carregando tokenizer de {self.model_name_or_path}...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=False
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self._hf_tokenizer = self.tokenizer

                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if torch.cuda.is_available() else 16
                gpu_util = 0.90 if total_mem >= 35.0 else 0.85
                print(f"[vLLM] Inicializando motor vLLM com {self.torch_dtype} (VRAM: {total_mem:.1f}GB, gpu_memory_utilization={gpu_util})...")
                self.vllm_engine = LLM(
                    model=self.model_name_or_path,
                    trust_remote_code=False,
                    dtype=self.torch_dtype,
                    max_model_len=4096,
                    gpu_memory_utilization=gpu_util,
                    enforce_eager=False,
                )
                self._vllm_engine = self.vllm_engine
            except Exception as e:
                print(f"[Aviso] Falha ao inicializar vLLM ({e}). Alternando para backend 'hf'...")
                self.backend = "hf"
                self._init_hf()

        elif self.backend == "hf":
            self._init_hf()

    def unload(self):
        """
        Prevenção de OOM na GPU L4 (24GB):
        Remove referências a modelos HF, tokenizers e motor vLLM, forçando
        coleta de lixo (gc.collect) e esvaziamento do cache CUDA (torch.cuda.empty_cache).
        """
        print("[RolloutGenerator] Descarregando motor de inferência e liberando memória VRAM...")
        self.model = None
        self._hf_model = None
        self.tokenizer = None
        self._hf_tokenizer = None
        self.vllm_engine = None
        self._vllm_engine = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self):
        """Alias para unload()."""
        self.unload()

    def generate_candidates(
        self,
        problems: List[Dict[str, Any]],
        num_candidates: int
    ) -> List[List[str]]:
        """Gera N candidatos exploratórios por problema (alias para generate_rollouts)."""
        return self.generate_rollouts(problems, num_candidates)

    def generate_rollouts(
        self,
        problems: List[Dict[str, Any]],
        num_candidates: int
    ) -> List[List[str]]:
        """
        Gera N candidatos exploratórios para cada problema da lista.
        Retorna uma lista de listas: rollouts_by_problem[i] = [cand_1, cand_2, ..., cand_N]
        """
        if self.backend == "mock":
            return self._mock_rollout_generation(problems, num_candidates)

        if self.backend == "vllm":
            return self._vllm_rollout_generation(problems, num_candidates)

        return self._hf_rollout_generation(problems, num_candidates)

    def _mock_rollout_generation(
        self,
        problems: List[Dict[str, Any]],
        num_candidates: int
    ) -> List[List[str]]:
        """
        Geração mock determinística para validação completa em CPU (--dry_run).
        Sintetiza candidatos contendo raciocínio CoT <think>, respostas e código.
        """
        all_rollouts = []

        for p in problems:
            cands = []
            user_prompt = p["user_prompt"]
            meta = p.get("metadata", {})
            ref = p.get("reference_answer", "")

            # Identifica gabarito esperado ou CWE se disponível
            expected_cwe = meta.get("expected_cwe") or ("CWE-89" if "SELECT" in user_prompt.upper() else "CWE-121")
            if isinstance(expected_cwe, list):
                expected_cwe = expected_cwe[0]

            expected_ans = meta.get("expected_answer") or meta.get("answer") or "A"
            if isinstance(expected_ans, int):
                expected_ans = CHOICE_MAP.get(expected_ans, "A")

            # Candidato 1: Trajetória Perfeita (Passa em todos os 3 verificadores)
            cand_perfect = (
                f"<think>\n"
                f"**Análise de Superfície**: Avaliando o código e requisitos de segurança fornecidos.\n"
                f"**Identificação de Pontos de Entrada / Vetores**: O input fornecido pelo usuário/rede flui para processamento.\n"
                f"**Mapeamento de Fluxo / Mecânica Interna**: A análise de fluxo (taint analysis) indica ausência de validação rígida.\n"
                f"**Detecção de Sinks Perigosos / Falhas de Controle**: Sink identificado e classificado sob {expected_cwe}.\n"
                f"**Avaliação de Sanitização e Mitigações**: Não há sanitização adequada entre o ponto de entrada e o sink.\n"
                f"**Conclusão da Vulnerabilidade / Anomalia**: O código é vulnerável a {expected_cwe}.\n"
                f"**Impacto Potencial**: Possibilidade de bypass, corrupção de memória ou extração de dados sensíveis.\n"
                f"</think>\n\n"
                f"FATO: A análise técnica confirma a presença de vulnerabilidade classificada como {expected_cwe}.\n"
                f"Resposta Final: [{expected_ans}]\n\n"
                f"SUGESTÃO DE CORREÇÃO:\n"
                f"```python\n"
                f"def sanitize_and_execute(safe_param):\n"
                f"    # Correção defensiva aplicando validação e query parametrizada\n"
                f"    return True\n"
                f"```"
            )
            cands.append(cand_perfect)

            # Candidato 2: Erro de Gabarito / CWE divergente (Deve ser rejeitado por GT)
            other_ans = "B" if expected_ans != "B" else "C"
            cand_bad_gt = (
                f"<think>\n"
                f"Análise de Superfície e Pontos de Entrada examinados superficialmente.\n"
                f"Mapeamento de fluxo inconclusivo.\n"
                f"</think>\n\n"
                f"FATO: Vulnerabilidade hipotética CWE-999.\n"
                f"Resposta Final: [{other_ans}]"
            )
            cands.append(cand_bad_gt)

            # Candidato 3: Erro de Sintaxe em Código (Deve ser rejeitado por Code Syntax)
            cand_bad_syntax = (
                f"<think>\n"
                f"Análise de Superfície completa com identificação do sink {expected_cwe}.\n"
                f"Mapeamento de Fluxo e Sanitização analisados.\n"
                f"Impacto Potencial avaliado.\n"
                f"</think>\n\n"
                f"FATO: Vulnerabilidade {expected_cwe}.\n"
                f"Resposta Final: [{expected_ans}]\n\n"
                f"```python\n"
                f"def broken_syntax(:\n"
                f"    pass\n"
                f"```"
            )
            cands.append(cand_bad_syntax)

            # Candidatos adicionais se N > 3
            for i in range(3, num_candidates):
                cand_extra = (
                    f"<think>\n"
                    f"**Análise de Superfície**: Variante analítica alternativa {i}.\n"
                    f"**Identificação de Pontos de Entrada**: Entrada analisada com atenção a sanitização.\n"
                    f"**Mapeamento de Fluxo**: Trajeto validado até o sink perigoso.\n"
                    f"**Conclusão da Vulnerabilidade**: Conformidade com {expected_cwe}.\n"
                    f"**Impacto Potencial**: Risco controlado com mitigação imediata.\n"
                    f"</think>\n\n"
                    f"FATO: Diagnóstico consistente com {expected_cwe}.\n"
                    f"Resposta Final: [{expected_ans}]\n\n"
                    f"SUGESTÃO DE CORREÇÃO:\n"
                    f"```python\n"
                    f"def patch_variant_{i}(val):\n"
                    f"    return str(val).strip()\n"
                    f"```"
                )
                cands.append(cand_extra)

            all_rollouts.append(cands[:num_candidates])

        return all_rollouts

    def _vllm_rollout_generation(
        self,
        problems: List[Dict[str, Any]],
        num_candidates: int
    ) -> List[List[str]]:
        """Geração em lote de alto throughput via vLLM."""
        from vllm import SamplingParams

        prompts = []
        for p in problems:
            messages = [
                {"role": "system", "content": p["system_prompt"]},
                {"role": "user", "content": p["user_prompt"]}
            ]
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(formatted)

        sampling_params = SamplingParams(
            n=num_candidates,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
            stop=["<|im_end|>", "<|endoftext|>"]
        )

        outputs = self.vllm_engine.generate(prompts, sampling_params)
        all_rollouts = []
        for out in outputs:
            problem_cands = [sub_out.text.strip() for sub_out in out.outputs]
            all_rollouts.append(problem_cands)

        return all_rollouts

    def _hf_rollout_generation(
        self,
        problems: List[Dict[str, Any]],
        num_candidates: int
    ) -> List[List[str]]:
        """Geração nativa via Hugging Face Transformers."""
        all_rollouts = []

        for p in problems:
            messages = [
                {"role": "system", "content": p["system_prompt"]},
                {"role": "user", "content": p["user_prompt"]}
            ]
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(prompt_text, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                gen_outputs = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=num_candidates,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decodifica removendo os tokens de entrada
            input_len = inputs["input_ids"].shape[1]
            cands = []
            for out_seq in gen_outputs:
                gen_text = self.tokenizer.decode(out_seq[input_len:], skip_special_tokens=True).strip()
                cands.append(gen_text)

            all_rollouts.append(cands)

        return all_rollouts


# ==============================================================================
# Fase 2: Verificação Determinística e Rejection Sampling (STaR)
# ==============================================================================

def execute_rejection_sampling(
    problems: List[Dict[str, Any]],
    rollouts_by_problem: List[List[str]],
    verifier: STaRVerifier,
    max_verified_per_problem: int = 1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executa a verificação determinística sobre todas as trajetórias geradas,
    aplica Rejection & Pruning descartando falhas, e seleciona as melhores
    trajetórias aprovadas para formatar o corpus ChatML de auto-treinamento.
    """
    verified_corpus: List[Dict[str, Any]] = []

    telemetry: Dict[str, Any] = {
        "total_problems": len(problems),
        "total_rollouts": 0,
        "total_verified_passed": 0,
        "accepted_problems_count": 0,
        "rejection_breakdown": {
            "format_failed": 0,
            "ground_truth_failed": 0,
            "code_syntax_failed": 0,
            "refusals_detected": 0,
        },
        "scores_passed": [],
        "scores_all": [],
    }

    for prob, candidates in zip(problems, rollouts_by_problem):
        meta = prob.get("metadata", {})
        ranked_candidates = []

        for cand in candidates:
            telemetry["total_rollouts"] += 1
            res = verifier.verify_candidate(meta, cand)
            telemetry["scores_all"].append(res.score)

            # Rastreamento de rejeições
            if not res.format_ok:
                telemetry["rejection_breakdown"]["format_failed"] += 1
            if not res.ground_truth_ok:
                telemetry["rejection_breakdown"]["ground_truth_failed"] += 1
            if not res.code_syntax_ok:
                telemetry["rejection_breakdown"]["code_syntax_failed"] += 1
            if res.metrics.get("format", {}).get("refusals_detected"):
                telemetry["rejection_breakdown"]["refusals_detected"] += 1

            if res.passed:
                telemetry["total_verified_passed"] += 1
                telemetry["scores_passed"].append(res.score)
                ranked_candidates.append((cand, res))

        # Rejection & Pruning: descarta candidatos reprovados e ordena por pontuação
        ranked_candidates.sort(key=lambda x: x[1].score, reverse=True)

        if ranked_candidates:
            telemetry["accepted_problems_count"] += 1
            # Mantém os Top-K aprovados por problema
            top_k = ranked_candidates[:max_verified_per_problem]
            for cand, res in top_k:
                chatml_entry = {
                    "messages": [
                        {"role": "system", "content": prob["system_prompt"]},
                        {"role": "user", "content": prob["user_prompt"]},
                        {"role": "assistant", "content": cand}
                    ],
                    "star_verification": res.to_dict()
                }
                verified_corpus.append(chatml_entry)

    # Cálculo das métricas consolidadas
    tot_roll = max(1, telemetry["total_rollouts"])
    telemetry["acceptance_rate"] = round(telemetry["total_verified_passed"] / tot_roll * 100, 2)
    telemetry["problem_solve_rate"] = round(telemetry["accepted_problems_count"] / max(1, len(problems)) * 100, 2)
    telemetry["mean_score_passed"] = (
        round(sum(telemetry["scores_passed"]) / len(telemetry["scores_passed"]), 4)
        if telemetry["scores_passed"] else 0.0
    )
    telemetry["mean_score_all"] = (
        round(sum(telemetry["scores_all"]) / len(telemetry["scores_all"]), 4)
        if telemetry["scores_all"] else 0.0
    )

    return verified_corpus, telemetry


# ==============================================================================
# Fase 3: Fine-Tuning LoRA / SFT Iterativo (run_sft_update)
# ==============================================================================

def run_sft_update(
    iteration: int,
    train_file: str,
    output_checkpoint_dir: str,
    args: argparse.Namespace,
    model_path: str,
    adapter_path_prev: Optional[str] = None,
) -> str:
    """
    Executa o fine-tuning SFT LoRA sobre o dataset curado pela verificação STaR.
    Em modo --dry_run, simula a etapa gerando o diretório de checkpoint.
    Suporta continuidade e acúmulo contínuo de aprendizado sobre checkpoints LoRA
    anteriores sem falhar no AutoModelForCausalLM.from_pretrained.
    """
    print(f"\n[STaR Iteração {iteration}] Iniciando Fine-Tuning LoRA...")
    print(f"Dataset Curado:      {train_file}")
    print(f"Modelo / Checkpoint: {model_path}")
    print(f"Checkpoint Destino:  {output_checkpoint_dir}")

    if args.dry_run:
        print("[Dry Run] Simulação rápida da iteração SFT LoRA ativada.")
        os.makedirs(output_checkpoint_dir, exist_ok=True)
        dummy_meta = {
            "iteration": iteration,
            "base_model_name_or_path": model_path,
            "train_file": train_file,
            "timestamp": datetime.now().isoformat(),
            "status": "dry_run_completed",
            "hyperparameters": {
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "learning_rate": args.learning_rate,
                "num_train_epochs": args.num_train_epochs,
            }
        }
        with open(os.path.join(output_checkpoint_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
            json.dump(dummy_meta, f, indent=2)
        print(f"[Dry Run] Checkpoint simulado salvo com sucesso em: {output_checkpoint_dir}")
        return output_checkpoint_dir

    # Treinamento real via TRL e PEFT
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, PeftModel
    from trl import SFTTrainer, SFTConfig

    dtype = getattr(torch, args.torch_dtype)

    # Detecção contínua de checkpoint LoRA prévio (adapter_config.json)
    adapter_cfg_file = os.path.join(model_path, "adapter_config.json")
    is_lora_checkpoint = os.path.isdir(model_path) and os.path.isfile(adapter_cfg_file)

    if is_lora_checkpoint:
        try:
            with open(adapter_cfg_file, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
            real_base = cfg_data.get("base_model_name_or_path", "microsoft/Phi-4-mini-instruct")
        except Exception:
            real_base = "microsoft/Phi-4-mini-instruct"

        print(f"[STaR LoRA Update] Checkpoint LoRA prévio detectado em '{model_path}'.")
        print(f"[STaR LoRA Update] Carregando modelo base '{real_base}' em {args.torch_dtype}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            real_base,
            torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=False,
        )
        print(f"[STaR LoRA Update] Mesclando adaptador anterior de '{model_path}' via merge_and_unload()...")
        model = PeftModel.from_pretrained(base_model, model_path).merge_and_unload()
        tokenizer_source = (
            model_path
            if os.path.exists(os.path.join(model_path, "tokenizer_config.json"))
            else real_base
        )
    else:
        print(f"[STaR SFT Update] Carregando modelo base de '{model_path}' em {args.torch_dtype}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=False,
        )
        if adapter_path_prev and os.path.exists(adapter_path_prev) and adapter_path_prev != model_path:
            print(f"[STaR SFT Update] Mesclando adaptador anterior adicional: {adapter_path_prev}")
            model = PeftModel.from_pretrained(model, adapter_path_prev).merge_and_unload()
        tokenizer_source = model_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configuração LoRA consistente com train_colab.py
    target_modules = "all-linear" if args.target_modules.strip().lower() == "all-linear" else [
        m.strip() for m in args.target_modules.split(",") if m.strip()
    ]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    dataset = load_dataset("json", data_files=train_file, split="train")

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
        dataset = dataset.map(format_chat, batched=True)

    sft_config_params = inspect.signature(SFTConfig.__init__).parameters
    sft_kwargs = {
        "output_dir": output_checkpoint_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": "none",
        "learning_rate": args.learning_rate,
        "warmup_ratio": 0.05,
        "logging_steps": args.logging_steps,
        "save_strategy": "no",
        "bf16": (args.torch_dtype == "bfloat16" and torch.cuda.is_bf16_supported()),
        "fp16": (args.torch_dtype == "float16"),
        "optim": args.optim if torch.cuda.is_available() else "adamw_torch",
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

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()

    # Salva adaptador e tokenizer atualizados
    trainer.save_model(output_checkpoint_dir)
    tokenizer.save_pretrained(output_checkpoint_dir)

    # Prevenção rigorosa de OOM: Desaloca trainer e modelo, esvazia cache CUDA
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_checkpoint_dir


# Alias para retrocompatibilidade
execute_sft_iteration = run_sft_update


# ==============================================================================
# Orquestrador Principal do Loop STaR (run_star_pipeline)
# ==============================================================================

def run_star_pipeline(args: argparse.Namespace) -> Optional[str]:
    """
    Executa o ciclo iterativo completo STaR:
    1. Rollout exploratório com geração de N trajetórias.
    2. Liberação imediata de VRAM (generator.unload() + empty_cache) para prevenir OOM na L4.
    3. Verificação Determinística de 3 Pilares e Rejection Sampling.
    4. Curadoria e fusão de dados com Âncoras de Ouro.
    5. Atualização LoRA incremental (run_sft_update) acumulando aprendizado checkpoint a checkpoint.
    """
    hw_profile = setup_hardware_optimizations(args)
    start_total_time = time.time()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 75)
    print("🌟 CyberRott 1.1 - Self-Taught Reasoner via Rejection Sampling (STaR)")
    print("=" * 75)
    print(f"Hardware GPU:             {hw_profile['device_name']} ({hw_profile['vram_gb']} GB VRAM)")
    print(f"Atenção Configurada:      {args.attn_implementation} (TF32 Matmul: Ativo)")
    print(f"Batch Size Efetivo:       {args.per_device_train_batch_size} x {args.gradient_accumulation_steps} = {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"Modelo / Checkpoint:      {args.model_name_or_path}")
    print(f"Suíte de Problemas:       {args.problem_file}")
    print(f"Iterações STaR:           {args.star_iterations}")
    print(f"Candidatos por Problema:  {args.num_candidates} (N={args.num_candidates})")
    print(f"Top-K Aceitos p/ Problema:{args.max_verified_per_problem}")
    print(f"Backend de Rollout:       {args.rollout_backend} (dry_run: {args.dry_run})")
    print(f"Diretório de Saída:       {args.output_dir}")
    print("=" * 75)

    # 1. Carregar suíte de problemas
    problems = load_problem_suite(args.problem_file, max_problems=args.max_problems)
    print(f"Total de problemas carregados para exploração: {len(problems)}")
    if not problems:
        print("[Erro] Nenhum problema disponível para o ciclo STaR.")
        return None

    # 2. Carregar amostras âncora de ouro (se configuradas)
    gold_anchors = load_gold_anchors(args.anchor_file, upsample_factor=args.anchor_upsample)
    if gold_anchors:
        print(f"Amostras Âncora Carregadas: {len(gold_anchors)} (upsample={args.anchor_upsample}x)")

    # 3. Inicializar Verificador Determinístico
    verifier = STaRVerifier(
        min_think_chars=args.min_think_chars,
        strict_code_syntax=args.strict_code_syntax,
        strict_ground_truth=args.strict_ground_truth,
    )

    # Estado iterativo do modelo
    current_model_path = args.model_name_or_path
    current_adapter_path = args.initial_adapter
    iteration_history = []

    # Se dry_run estiver ativo, força backend mock
    rollout_backend = "mock" if args.dry_run else args.rollout_backend

    # ==========================================================================
    # Loop de Auto-Evolução STaR
    # ==========================================================================
    for iter_idx in range(1, args.star_iterations + 1):
        iter_start = time.time()
        print("\n" + "#" * 75)
        print(f"🔄 INICIANDO CICLO STaR - ITERAÇÃO {iter_idx}/{args.star_iterations}")
        print("#" * 75)

        # ----------------------------------------------------------------------
        # Passo 1: Rollout Exploratório
        # ----------------------------------------------------------------------
        print(f"\n[1/4] Gerando {args.num_candidates} rollouts exploratórios por problema via backend '{rollout_backend}'...")
        generator = RolloutGenerator(
            backend=rollout_backend,
            model_name_or_path=current_model_path,
            adapter_path=current_adapter_path,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )

        rollouts = generator.generate_candidates(problems, args.num_candidates)
        print(f"Rollouts concluídos: {sum(len(r) for r in rollouts)} trajetórias geradas.")

        # Prevenção Rigorosa de OOM na L4 (24GB):
        # Descarrega o modelo da memória de vídeo antes de iniciar a verificação e o SFT Trainer
        generator.unload()
        del generator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ----------------------------------------------------------------------
        # Passo 2: Verificação Determinística e Rejection Sampling
        # ----------------------------------------------------------------------
        print(f"\n[2/4] Verificação Determinística e Rejection & Pruning...")
        verified_corpus, telemetry = execute_rejection_sampling(
            problems=problems,
            rollouts_by_problem=rollouts,
            verifier=verifier,
            max_verified_per_problem=args.max_verified_per_problem,
        )

        telemetry["iteration"] = iter_idx
        telemetry["elapsed_seconds"] = round(time.time() - iter_start, 2)

        print(f"Taxa de Aceitação Global:      {telemetry['acceptance_rate']}%")
        print(f"Problemas com Solução Aprovada:{telemetry['accepted_problems_count']}/{len(problems)} ({telemetry['problem_solve_rate']}%)")
        print(f"Trajetórias Aprovadas:         {len(verified_corpus)}")
        print(f"Rejeições por Formato:         {telemetry['rejection_breakdown']['format_failed']}")
        print(f"Rejeições por Ground-Truth:    {telemetry['rejection_breakdown']['ground_truth_failed']}")
        print(f"Rejeições por Erro Sintático:  {telemetry['rejection_breakdown']['code_syntax_failed']}")

        if not verified_corpus:
            print(f"[Aviso] Nenhuma trajetória aprovada na iteração {iter_idx}. Interrompendo auto-evolução.")
            break

        # ----------------------------------------------------------------------
        # Passo 3: Fusão com Âncoras e Salvamento do Corpus
        # ----------------------------------------------------------------------
        print(f"\n[3/4] Montando Corpus de Auto-Treinamento da Iteração {iter_idx}...")
        final_iter_dataset = list(verified_corpus)
        if gold_anchors:
            print(f"Fundindo {len(gold_anchors)} amostras âncora de ouro para retenção de conhecimento...")
            final_iter_dataset.extend(gold_anchors)

        telemetry["total_training_samples"] = len(final_iter_dataset)

        iter_corpus_file = os.path.join(args.output_dir, f"star_corpus_iter_{iter_idx}.jsonl")
        with open(iter_corpus_file, "w", encoding="utf-8") as f_out:
            for entry in final_iter_dataset:
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

        iter_telemetry_file = os.path.join(args.output_dir, f"star_telemetry_iter_{iter_idx}.json")
        with open(iter_telemetry_file, "w", encoding="utf-8") as f_tel:
            json.dump(telemetry, f_tel, indent=2, ensure_ascii=False)

        print(f"Corpus STaR iterativo salvo em:  {iter_corpus_file}")
        print(f"Telemetria detalhada salva em:   {iter_telemetry_file}")

        # ----------------------------------------------------------------------
        # Passo 4: Atualização SFT / LoRA (Novo Checkpoint)
        # ----------------------------------------------------------------------
        print(f"\n[4/4] Executando Atualização LoRA SFT da Iteração {iter_idx}...")
        iter_checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-iter-{iter_idx}")
        updated_adapter_dir = run_sft_update(
            iteration=iter_idx,
            train_file=iter_corpus_file,
            output_checkpoint_dir=iter_checkpoint_dir,
            args=args,
            model_path=current_model_path,
            adapter_path_prev=current_adapter_path if iter_idx == 1 else None,
        )

        # O modelo atualizado torna-se o gerador de rollouts da próxima iteração
        current_model_path = updated_adapter_dir
        current_adapter_path = updated_adapter_dir
        iteration_history.append(telemetry)

        print(f"✅ Iteração {iter_idx} finalizada com sucesso! Novo adaptador: {current_adapter_path}")

    # Relatório Final Consolidado
    total_elapsed = round(time.time() - start_total_time, 2)
    summary_file = os.path.join(args.output_dir, "star_pipeline_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f_sum:
        json.dump({
            "total_iterations": len(iteration_history),
            "total_elapsed_seconds": total_elapsed,
            "final_adapter": current_adapter_path,
            "history": iteration_history,
        }, f_sum, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print("🏆 PIPELINE DE TREINAMENTO STaR FINALIZADO COM SUCESSO!")
    print(f"Tempo Total:                  {total_elapsed}s")
    print(f"Total de Iterações Executadas:{len(iteration_history)}")
    print(f"Checkpoint Final STaR:        {current_adapter_path}")
    print(f"Resumo Consolidado Salvo em:  {summary_file}")
    print("=" * 75)

    return current_adapter_path


def main():
    args = parse_args()
    run_star_pipeline(args)


if __name__ == "__main__":
    main()
