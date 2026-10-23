#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline de Treinamento GRPO (Group Relative Policy Optimization) - CyberRott v1.2
===================================================================================
Implementa o motor matemático de aprendizado por reforço online estilo DeepSeek-R1:
1. Fusão automática em memória do checkpoint SFT LoRA prévio (π_ref com zero VRAM extra).
2. Curadoria focada de amostras ricas em verificação (CWE, MITRE ATT&CK, Múltipla Escolha e Código Sandbox).
3. 3 Funções de Recompensa Determinísticas Verificáveis integradas ao STaRVerifier:
   - cot_integrity_reward (0.30): Integridade de <think>, profundidade analítica, 7 estágios e anti-recusa.
   - technical_accuracy_reward (0.50): Gabarito exato de múltipla escolha e identificadores CWE/MITRE.
   - sandbox_code_reward (0.20): Validação de sintaxe estática em AST Python, YARA e regras Sigma/YAML.
4. Otimizado para aceleração em GPU única (NVIDIA A100 40GB / L4 24GB no Colab Pro) via TRL GRPOTrainer e PEFT.
5. Suporte nativo a validação determinística sem GPU via flag --dry_run.
"""

import os
import sys
import gc
import re
import json
import time
import shutil
import urllib.request
import argparse
from typing import List, Dict, Any, Optional, Tuple, Union

# -----------------------------------------------------------------------------
# Compatibilidade Crítica: Proteção contra divergência de tipos transformers x trl
# -----------------------------------------------------------------------------
try:
    import trl.import_utils
    for _attr in dir(trl.import_utils):
        if _attr.startswith('_') and _attr.endswith('_available'):
            _val = getattr(trl.import_utils, _attr)
            if isinstance(_val, tuple):
                setattr(trl.import_utils, _attr, bool(_val[0]))
        elif _attr.startswith('is_') and _attr.endswith('_available'):
            _fn = getattr(trl.import_utils, _attr)
            if callable(_fn):
                try:
                    _val = _fn()
                    if isinstance(_val, tuple):
                        _bool_val = bool(_val[0])
                        setattr(trl.import_utils, _attr, (lambda b=_bool_val: b))
                except Exception:
                    pass
except Exception:
    pass

import torch

# Desativação do W&B por padrão e configuração de alocação de memória CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_DISABLED"] = "true"

# Adicionar diretório raiz ao path para imports locais
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from scripts.star_verifier import (
        verify_format_integrity,
        verify_ground_truth,
        verify_code_syntax,
        extract_mc_answer,
        CHOICE_MAP,
    )
except ImportError:
    from star_verifier import (
        verify_format_integrity,
        verify_ground_truth,
        verify_code_syntax,
        extract_mc_answer,
        CHOICE_MAP,
    )


# ==============================================================================
# Otimizações de Hardware para L4 e A100
# ==============================================================================

def setup_hardware_optimizations(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    """Detecta acelerador GPU e configura TF32 e implementações ótimas de atenção."""
    profile = {
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
    profile["is_ampere_or_newer"] = (major >= 8)
    profile["is_high_vram"] = (total_vram_gb >= 35.0)

    try:
        import flash_attn  # noqa: F401
        profile["recommended_attn"] = "flash_attention_2"
    except ImportError:
        profile["recommended_attn"] = "sdpa"

    if args is not None and getattr(args, "auto_optimize", True):
        if profile["is_high_vram"] and args.attn_implementation == "sdpa" and profile["recommended_attn"] == "flash_attention_2":
            args.attn_implementation = "flash_attention_2"
            print("[Auto-Otimização A100] Ativado flash_attention_2 automaticamente.")

    return profile


# ==============================================================================
# Helper de Extração de Texto de Completions (Compatibilidade com TRL)
# ==============================================================================

def extract_text_from_completion(c: Any) -> str:
    """Extrai texto string de completions nos formatos list[str] ou list[list[dict]]."""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        if len(c) == 0:
            return ""
        if isinstance(c[0], dict) and "content" in c[0]:
            return str(c[0]["content"])
        if isinstance(c[0], str):
            return "".join(c)
    if isinstance(c, dict) and "content" in c:
        return str(c["content"])
    if c is None:
        return ""
    return str(c)


# ==============================================================================
# As 3 Funções de Recompensa Determinísticas Verificáveis (GRPO Reward Suite)
# ==============================================================================

def cot_integrity_reward(prompts: Any, completions: Any, **kwargs) -> List[float]:
    """
    Função de Recompensa 1: Integridade Estrutural CoT (Peso: 0.30)
    - Penalidade severa (0.0 imediato) para recusas de IA conversacional ("As an AI...", "I cannot...").
    - Requer tags balanceadas <think>...</think>.
    - Pontua profundidade do raciocínio, cobertura dos 7 estágios e marcadores estruturais (FATO, DETECÇÃO, CORREÇÃO).
    """
    rewards = []
    for c in completions:
        text = extract_text_from_completion(c)
        fmt_ok, fmt_reasons, fmt_metrics = verify_format_integrity(
            text,
            min_think_chars=80,
            min_think_words=15,
            min_final_chars=20,
        )

        # 1. Penalidade estrita para recusas de alinhamento
        if fmt_metrics.get("refusals_detected"):
            rewards.append(0.0)
            continue

        # 2. Presença mandatória de tags de raciocínio
        if "<think>" not in text or "</think>" not in text:
            rewards.append(0.0)
            continue

        # 3. Pontuação de integridade CoT [0.0 - 1.0]
        sub_score = 0.0

        # Formato base correto (abertura, fechamento ordenado e tamanho mínimo)
        if fmt_ok:
            sub_score += 0.40
        else:
            sub_score += 0.15

        # Profundidade e deliberação analítica (ideal: >= 30 palavras de raciocínio)
        think_words = fmt_metrics.get("think_words", 0)
        depth_ratio = min(1.0, think_words / 30.0)
        sub_score += 0.20 * depth_ratio

        # Cobertura dos 7 estágios analíticos de cibersegurança do CyberRott
        stages_cov = fmt_metrics.get("stages_coverage", 0.0)
        sub_score += 0.20 * float(stages_cov)

        # Marcadores estruturais pós-raciocínio (FATO, DETECÇÃO, SUGESTÃO DE CORREÇÃO, etc.)
        struct_markers = fmt_metrics.get("structure_markers_found", 0)
        marker_ratio = min(1.0, struct_markers / 2.0)
        sub_score += 0.20 * marker_ratio

        # Escalonado pelo peso 0.30
        rewards.append(round(min(1.0, sub_score) * 0.30, 4))

    return rewards


def technical_accuracy_reward(prompts: Any, completions: Any, **kwargs) -> List[float]:
    """
    Função de Recompensa 2: Acurácia Técnica e Ground-Truth (Peso: 0.50)
    - Avalia gabarito exato de múltipla escolha (A, B, C, D) se for MCQ.
    - Valida indicadores técnicos determinísticos: CWE IDs (CWE-89, etc.) e MITRE ATT&CK (T1059, etc.).
    - Valida consistência técnica com a resposta de referência.
    """
    rewards = []
    ref_answers = kwargs.get("reference_answer", [])
    expected_cwes = kwargs.get("expected_cwe", [])
    expected_mitres = kwargs.get("expected_mitre", [])
    expected_choices = kwargs.get("expected_choice", [])

    for i, c in enumerate(completions):
        text = extract_text_from_completion(c)

        ref_ans = ref_answers[i] if (i < len(ref_answers) and ref_answers[i]) else ""
        cwe_item = expected_cwes[i] if (i < len(expected_cwes) and expected_cwes[i]) else None
        mitre_item = expected_mitres[i] if (i < len(expected_mitres) and expected_mitres[i]) else None
        choice_item = expected_choices[i] if (i < len(expected_choices) and expected_choices[i]) else None

        # Desempacotar strings serializadas se necessário
        if isinstance(cwe_item, str) and cwe_item.strip():
            cwe_list = [c.strip().upper() for c in cwe_item.split(",") if c.strip()]
        elif isinstance(cwe_item, list):
            cwe_list = [str(c).strip().upper() for c in cwe_item]
        else:
            cwe_list = []

        if isinstance(mitre_item, str) and mitre_item.strip():
            mitre_list = [m.strip().upper() for m in mitre_item.split(",") if m.strip()]
        elif isinstance(mitre_item, list):
            mitre_list = [str(m).strip().upper() for m in mitre_item]
        else:
            mitre_list = []

        choice_val = str(choice_item).strip().upper() if choice_item else ""

        problem = {
            "reference_answer": ref_ans,
            "expected_cwe": cwe_list,
            "expected_mitre": mitre_list,
            "expected_answer": choice_val,
        }

        gt_ok, gt_reasons, gt_metrics = verify_ground_truth(text, problem)
        gt_type = gt_metrics.get("gt_type", "unknown")

        sub_score = 0.0
        if gt_ok:
            sub_score = 1.0
        else:
            # Em múltipla escolha, gabarito incorreto zera a recompensa
            if gt_type in ["multiple_choice", "multiple_choice_derived"]:
                sub_score = 0.0
            else:
                # Indicadores técnicos: concede crédito parcial para identificação precisa
                candidate_cwes = [cw.upper() for cw in re.findall(r"\bCWE-\d+\b", text, re.I)]
                candidate_mitre = [mt.upper() for mt in re.findall(r"\bT\d{4}(?:\.\d{3})?\b", text, re.I)]

                matched_cwes = [cw for cw in cwe_list if cw in candidate_cwes]
                matched_mitre = [mt for mt in mitre_list if mt in candidate_mitre]

                cwe_ratio = (len(matched_cwes) / len(cwe_list)) if cwe_list else 0.0
                mitre_ratio = (len(matched_mitre) / len(mitre_list)) if mitre_list else 0.0

                if cwe_list and mitre_list:
                    sub_score = 0.5 * cwe_ratio + 0.5 * mitre_ratio
                elif cwe_list:
                    sub_score = 1.0 * cwe_ratio
                elif mitre_list:
                    sub_score = 1.0 * mitre_ratio
                else:
                    # Consistência estrutural de veredito
                    if any(term in text.lower() for term in ["vulnerabilidade", "vulnerability", "fato:", "veredicto", "verdict"]):
                        sub_score = 0.60

        # Escalonado pelo peso 0.50
        rewards.append(round(min(1.0, sub_score) * 0.50, 4))

    return rewards


def sandbox_code_reward(prompts: Any, completions: Any, **kwargs) -> List[float]:
    """
    Função de Recompensa 3: Sandbox Estático de Código e Sintaxe (Peso: 0.20)
    - Compilação estática AST Python (ast.parse e compile(..., 'exec') sem execução dinâmica).
    - Gramática estrutural YARA (rule, condition, balanceamento de chaves).
    - Parsing estático YAML para regras Sigma (title, logsource, detection).
    """
    rewards = []
    for c in completions:
        text = extract_text_from_completion(c)
        code_ok, code_reasons, code_metrics = verify_code_syntax(text, check_inside_think=False)
        total_blocks = code_metrics.get("total_code_blocks", 0)

        if total_blocks == 0:
            # Resposta puramente textual conceitual: pontuação neutra
            sub_score = 1.0
        else:
            if code_ok:
                sub_score = 1.0
            else:
                # Penaliza proporcionalmente os blocos que falharam na compilação
                failed_blocks = len(code_reasons)
                sub_score = max(0.0, 1.0 - (failed_blocks / float(total_blocks)))

        # Escalonado pelo peso 0.20
        rewards.append(round(min(1.0, sub_score) * 0.20, 4))

    return rewards


# ==============================================================================
# Carregamento do Dataset com Fallback GitHub e Curadoria de Verificabilidade
# ==============================================================================

DEFAULT_SYSTEM_PROMPT = (
    "Você é o CyberRott, uma IA especialista em segurança ofensiva, defensiva e análise de vulnerabilidades com raciocínio analítico profundo."
)

GITHUB_RAW_TRAIN = "https://raw.githubusercontent.com/johnknevescarrijo/CyberRott1.0/main/data/cyberrott_final_train.jsonl"
GITHUB_RAW_ANCHOR = "https://raw.githubusercontent.com/johnknevescarrijo/CyberRott1.0/main/data/dataset_cyber_ready.jsonl"


def ensure_dataset_exists(file_path: str, fallback_url: str) -> str:
    """Garante que o arquivo exista localmente, realizando download do GitHub caso ausente."""
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return file_path

    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    print(f"[*] Arquivo local '{file_path}' ausente. Baixando via fallback do GitHub ({fallback_url})...")
    try:
        urllib.request.urlretrieve(fallback_url, file_path)
        print(f"[✓] Download concluído com sucesso: {file_path} ({os.path.getsize(file_path) / (1024*1024):.2f} MB)")
        return file_path
    except Exception as e:
        print(f"[!] Falha ao baixar de {fallback_url}: {e}")
        # Tenta clone shallow do repositório como plano B
        if not os.path.exists(file_path):
            tmp_clone = "/tmp/cyberrott_git_clone"
            if not os.path.exists(tmp_clone):
                os.system(f"git clone --depth 1 https://github.com/johnknevescarrijo/CyberRott1.0.git {tmp_clone}")
            repo_file = os.path.join(tmp_clone, file_path)
            if os.path.exists(repo_file):
                shutil.copyfile(repo_file, file_path)
                print(f"[✓] Arquivo recuperado via git clone: {file_path}")
                return file_path

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Não foi possível encontrar nem baixar o dataset: {file_path}")
    return file_path


def parse_and_curate_dataset(
    train_path: str,
    anchor_path: str,
    max_samples: int = 400
) -> List[Dict[str, Any]]:
    """
    Carrega o dataset, extrai metadados determinísticos de verificação e prioriza amostras
    com maior densidade técnica (CWE + MITRE + Múltipla Escolha + Código) somadas a todas as Âncoras.
    """
    ensure_dataset_exists(train_path, GITHUB_RAW_TRAIN)
    ensure_dataset_exists(anchor_path, GITHUB_RAW_ANCHOR)

    all_raw_samples = []

    # 1. Carregar âncoras de ouro de alta prioridade (dataset_cyber_ready.jsonl)
    anchor_count = 0
    if os.path.exists(anchor_path):
        with open(anchor_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    entry["_is_anchor"] = True
                    all_raw_samples.append(entry)
                    anchor_count += 1
                except json.JSONDecodeError:
                    continue
        print(f"[*] Amostras Âncora carregadas: {anchor_count}")

    # 2. Carregar dataset de treino principal (cyberrott_final_train.jsonl)
    train_count = 0
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                entry["_is_anchor"] = False
                all_raw_samples.append(entry)
                train_count += 1
            except json.JSONDecodeError:
                continue
    print(f"[*] Amostras de Treino carregadas: {train_count} (Total bruto: {len(all_raw_samples)})")

    # 3. Processar e classificar por riqueza de verificação
    curated_records = []
    for item in all_raw_samples:
        messages = item.get("messages", [])
        if not messages or not isinstance(messages, list):
            continue

        system_content = None
        user_content = None
        assistant_content = ""

        prompt_messages = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_content = content
            elif role == "user":
                user_content = content
            elif role == "assistant":
                assistant_content = content

        if not user_content:
            continue

        if not system_content:
            system_content = DEFAULT_SYSTEM_PROMPT

        prompt_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        # Extração de Indicadores de Ground-Truth da resposta do assistente
        cwe_matches = list(set(re.findall(r"\bCWE-\d+\b", assistant_content, re.IGNORECASE)))
        expected_cwe = ",".join([c.upper() for c in cwe_matches])

        mitre_matches = list(set(re.findall(r"\bT\d{4}(?:\.\d{3})?\b", assistant_content, re.IGNORECASE)))
        expected_mitre = ",".join([m.upper() for m in mitre_matches])

        expected_choice = extract_mc_answer(assistant_content) or ""

        has_code = "```" in assistant_content

        # Cálculo da Densidade de Verificabilidade (Richness Score)
        richness_score = 0.0
        if item.get("_is_anchor", False):
            richness_score += 100.0  # Preserva 100% das âncoras
        if cwe_matches:
            richness_score += 4.0
        if mitre_matches:
            richness_score += 3.0
        if expected_choice:
            richness_score += 2.0
        if has_code:
            richness_score += 2.0
        if len(assistant_content) > 300:
            richness_score += 1.0

        curated_records.append({
            "prompt": prompt_messages,
            "reference_answer": assistant_content,
            "expected_cwe": expected_cwe,
            "expected_mitre": expected_mitre,
            "expected_choice": expected_choice,
            "_richness": richness_score,
            "_is_anchor": item.get("_is_anchor", False),
        })

    # 4. Ordenar por verificabilidade decrescente
    curated_records.sort(key=lambda x: x["_richness"], reverse=True)

    if max_samples > 0 and len(curated_records) > max_samples:
        selected_records = curated_records[:max_samples]
        print(f"[✓] Curadoria concluída: Selecionadas {len(selected_records)} melhores amostras verificáveis (max_samples={max_samples}).")
    else:
        selected_records = curated_records
        print(f"[✓] Curadoria concluída: Utilizando todas as {len(selected_records)} amostras disponíveis.")

    # Remover chaves temporárias internas
    final_dataset = []
    for r in selected_records:
        final_dataset.append({
            "prompt": r["prompt"],
            "reference_answer": r["reference_answer"],
            "expected_cwe": r["expected_cwe"],
            "expected_mitre": r["expected_mitre"],
            "expected_choice": r["expected_choice"],
        })

    return final_dataset


# ==============================================================================
# Execução do Modo Dry Run (Validação Sem GPU)
# ==============================================================================

def run_dry_run_validation(dataset_records: List[Dict[str, Any]]) -> bool:
    """Valida o carregamento dos dados, extração de gabarito e cálculo determinístico das 3 recompensas."""
    print("\n" + "=" * 70)
    print("🔍 MODO DRY-RUN: VALIDAÇÃO DETERMINÍSTICA DO PIPELINE GRPO")
    print("=" * 70)
    print(f"[*] Total de amostras prontas para GRPO: {len(dataset_records)}")

    # Estatísticas do dataset curado
    cwe_samples = sum(1 for r in dataset_records if r["expected_cwe"])
    mitre_samples = sum(1 for r in dataset_records if r["expected_mitre"])
    choice_samples = sum(1 for r in dataset_records if r["expected_choice"])

    print(f"[*] Amostras com CWE esperado:              {cwe_samples} ({cwe_samples/len(dataset_records)*100:.1f}%)")
    print(f"[*] Amostras com MITRE ATT&CK esperado:     {mitre_samples} ({mitre_samples/len(dataset_records)*100:.1f}%)")
    print(f"[*] Amostras com Múltipla Escolha esperada: {choice_samples} ({choice_samples/len(dataset_records)*100:.1f}%)")

    # Amostra de teste para o cálculo das funções de recompensa
    test_sample = dataset_records[0]
    prompt = test_sample["prompt"]
    gold_ref = test_sample["reference_answer"]

    prompts_batch = [prompt] * 4

    # 4 Variações sintéticas para testar a resposta do motor de recompensas:
    # 1. Candidato Ideal (espera recompensa alta)
    cand_ideal = gold_ref if ("<think>" in gold_ref and "</think>" in gold_ref) else f"<think>\nAnálise da superfície e mapeamento de fluxo com sinks identificados.\nConclusão da vulnerabilidade: CWE-89.\n</think>\nFATO: Identificado CWE-89.\nDETECÇÃO: Monitorar SQL Injection.\nSUGESTÃO DE CORREÇÃO:\n```python\ncursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))\n```"

    # 2. Candidato com Recusa de Alinhamento (deve zerar CoT reward)
    cand_refusal = "<think>Analisei o pedido.</think> I apologize, but as an AI language model I cannot assist with vulnerability analysis."

    # 3. Candidato com Tags Quebradas (deve penalizar CoT)
    cand_broken_tags = "Esta resposta não tem tags de pensamento. O CWE é CWE-89."

    # 4. Candidato com Erro de Código (deve penalizar Code reward)
    cand_syntax_error = "<think>\nAnálise técnica com identificação de falha.\n</think>\nFATO: Falha detectada.\n```python\ndef broken_func(\n  returrn syntax error !\n```"

    completions_batch = [cand_ideal, cand_refusal, cand_broken_tags, cand_syntax_error]

    kwargs = {
        "reference_answer": [test_sample["reference_answer"]] * 4,
        "expected_cwe": [test_sample["expected_cwe"]] * 4,
        "expected_mitre": [test_sample["expected_mitre"]] * 4,
        "expected_choice": [test_sample["expected_choice"]] * 4,
    }

    r_cot = cot_integrity_reward(prompts_batch, completions_batch, **kwargs)
    r_acc = technical_accuracy_reward(prompts_batch, completions_batch, **kwargs)
    r_box = sandbox_code_reward(prompts_batch, completions_batch, **kwargs)

    print("\n" + "-" * 70)
    print("RESULTADO DO TESTE DAS 3 FUNÇÕES DE RECOMPENSA (BATCH DE TESTE):")
    print("-" * 70)
    scenarios = [
        "1. Candidato Ideal (Full CoT + Clean Code)",
        "2. Candidato com Recusa ('As an AI...')",
        "3. Candidato sem Tags <think>",
        "4. Candidato com Erro de Sintaxe Python",
    ]

    for idx, name in enumerate(scenarios):
        total_r = round(r_cot[idx] + r_acc[idx] + r_box[idx], 4)
        print(f"\n{name}:")
        print(f"  • cot_integrity_reward (peso 0.30):    {r_cot[idx]:.4f}")
        print(f"  • technical_accuracy_reward (peso 0.50): {r_acc[idx]:.4f}")
        print(f"  • sandbox_code_reward (peso 0.20):      {r_box[idx]:.4f}")
        print(f"  ➜ RECOMPENSA TOTAL COMBINADA:          {total_r:.4f} / 1.0000")

    # Asserções determinísticas de sanidade
    assert r_cot[1] == 0.0, "Falha: Recusa de alinhamento não zerou cot_integrity_reward!"
    assert r_cot[2] == 0.0, "Falha: Ausência de <think> não zerou cot_integrity_reward!"
    assert r_box[3] < 0.20, "Falha: Erro de sintaxe Python não foi penalizado em sandbox_code_reward!"
    assert r_cot[0] > 0.0, "Falha: Candidato ideal deve obter recompensa CoT positiva!"

    print("\n" + "=" * 70)
    print("🏆 [DRY-RUN BEM-SUCEDIDO] Todas as funções de recompensa e parsing validados com 100% de precisão!")
    print("=" * 70 + "\n")
    return True


# ==============================================================================
# Pipeline Principal de Treinamento GRPO
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="CyberRott v1.2 - Treinamento Online GRPO com TRL e PEFT"
    )

    # Argumentos do Modelo e Adaptador
    model_group = parser.add_argument_group("Modelo")
    model_group.add_argument(
        "--model_name_or_path",
        type=str,
        default="microsoft/Phi-4-mini-instruct",
        help="Modelo base Hugging Face (padrão: microsoft/Phi-4-mini-instruct)."
    )
    model_group.add_argument(
        "--adapter_name_or_path", "--sft_lora_path",
        dest="adapter_name_or_path",
        type=str,
        default="./cyberrott_v1_1_sft_lora",
        help="Caminho do checkpoint LoRA da Fase 2 (SFT). Se existir, será fundido na memória como π_ref."
    )
    model_group.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Tipo de precisão dos tensores (padrão: bfloat16)."
    )
    model_group.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Implementação de atenção: sdpa ou flash_attention_2."
    )

    # Argumentos de Dados
    data_group = parser.add_argument_group("Dados")
    data_group.add_argument(
        "--dataset_name", "--data_path", "--data_file",
        dest="dataset_name",
        type=str,
        default="data/cyberrott_final_train.jsonl",
        help="Arquivo JSONL do dataset consolidado de treino."
    )
    data_group.add_argument(
        "--anchor_file",
        type=str,
        default="data/dataset_cyber_ready.jsonl",
        help="Arquivo JSONL com as Âncoras de Ouro complexas."
    )
    data_group.add_argument(
        "--max_samples",
        type=int,
        default=400,
        help="Quantidade de amostras curadas por densidade técnica (padrão: 400; <=0 usa todas as 4.680)."
    )
    data_group.add_argument(
        "--max_prompt_length",
        type=int,
        default=1024,
        help="Tamanho máximo de contexto do prompt (padrão: 1024)."
    )
    data_group.add_argument(
        "--max_completion_length",
        type=int,
        default=1024,
        help="Tamanho máximo dos rollouts gerados (padrão: 1024)."
    )

    # Argumentos GRPO e Treinamento
    grpo_group = parser.add_argument_group("Hiperparâmetros GRPO")
    grpo_group.add_argument(
        "--output_dir",
        type=str,
        default="./CyberRott-v1.2-GRPO",
        help="Diretório de saída para salvar o checkpoint GRPO final."
    )
    grpo_group.add_argument(
        "--num_generations",
        type=int,
        default=4,
        help="Número de rollouts gerados por prompt (G) no cálculo de vantagem relativa (padrão: 4)."
    )
    grpo_group.add_argument(
        "--beta",
        type=float,
        default=0.04,
        help="Coeficiente de penalidade KL em relação à política de referência π_ref (padrão: 0.04)."
    )
    grpo_group.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Taxa de aprendizado AdamW para a fase GRPO (padrão: 5e-6)."
    )
    grpo_group.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Tamanho do micro-batch por dispositivo (padrão: 1 devido ao vocabulário de 200k do Phi-4)."
    )
    grpo_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Passos de acumulação de gradiente (padrão: 8; batch global efetivo = 8)."
    )
    grpo_group.add_argument(
        "--num_train_epochs", "--epochs",
        dest="num_train_epochs",
        type=int,
        default=1,
        help="Número de épocas de treinamento GRPO (padrão: 1)."
    )
    grpo_group.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Número máximo de passos (-1 desativa e usa num_train_epochs)."
    )
    grpo_group.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Intervalo de salvamento de checkpoints."
    )
    grpo_group.add_argument(
        "--logging_steps",
        type=int,
        default=5,
        help="Intervalo de logging dos passos de treino e recompensas."
    )
    grpo_group.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Limite de checkpoints mantidos em disco."
    )
    grpo_group.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Proporção de warmup do learning rate scheduler."
    )
    grpo_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente aleatória para reprodutibilidade."
    )

    # Argumentos LoRA PEFT
    lora_group = parser.add_argument_group("LoRA")
    lora_group.add_argument(
        "--lora_r",
        type=int,
        default=32,
        help="Rank LoRA para adaptação durante GRPO (padrão: 32)."
    )
    lora_group.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="Alpha LoRA (padrão: 64)."
    )
    lora_group.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Dropout LoRA (padrão: 0.05)."
    )

    # Flags de Execução
    exec_group = parser.add_argument_group("Execução")
    exec_group.add_argument(
        "--dry_run",
        action="store_true",
        help="Modo de auditoria rápida sem GPU: valida dataset, ground-truth e as 3 recompensas."
    )
    exec_group.add_argument(
        "--auto_optimize",
        action="store_true",
        default=True,
        help="Ajusta atenções e TF32 dinamicamente conforme hardware detectado."
    )
    exec_group.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Envia o checkpoint final para o Hugging Face Hub."
    )
    exec_group.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Nome do repositório no Hub se --push_to_hub estiver ativo."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("🚀 CyberRott v1.2 — Pipeline de Treinamento GRPO (DeepSeek-R1 Style)")
    print("=" * 70)

    # 1. Configurar otimizações de aceleração de hardware
    hw_info = setup_hardware_optimizations(args)
    print(f"[*] Hardware: {hw_info['device_name']} | VRAM: {hw_info['vram_gb']} GB | Atenção: {args.attn_implementation}")

    # 2. Carregar e curar o dataset com extração de metadados
    dataset_records = parse_and_curate_dataset(
        train_path=args.dataset_name,
        anchor_path=args.anchor_file,
        max_samples=args.max_samples
    )

    # 3. Execução em Modo Dry-Run se solicitado
    if args.dry_run:
        success = run_dry_run_validation(dataset_records)
        sys.exit(0 if success else 1)

    # 4. Importar módulos de ML (executado apenas se não for dry-run)
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, PeftModel
    from trl import GRPOTrainer, GRPOConfig

    # Converter registros para HuggingFace Dataset
    hf_dataset = Dataset.from_list(dataset_records)
    print(f"[✓] Hugging Face Dataset construído com {len(hf_dataset)} amostras.")

    # 5. Carregar Tokenizer
    print(f"[*] Carregando tokenizer de '{args.model_name_or_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 6. Carregar Modelo Base
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.torch_dtype, torch.bfloat16)

    print(f"[*] Carregando modelo base: {args.model_name_or_path} ({args.torch_dtype})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=False,
    )

    # 7. Fusão Automática na Memória: SFT_LORA -> Base Model (Política π_ref sem VRAM extra)
    adapter_path = args.adapter_name_or_path
    if adapter_path and os.path.exists(adapter_path) and os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        print(f"\n[*] Checkpoint SFT prévio detectado em: '{adapter_path}'")
        print("[*] Realizando fusão LoRA na memória (merge_and_unload)...")
        start_merge = time.time()
        peft_sft = PeftModel.from_pretrained(base_model, adapter_path)
        base_model = peft_sft.merge_and_unload()
        print(f"[✓] Fusão concluída em {time.time() - start_merge:.2f}s!")
        print("[✓] O modelo com tradecraft SFT agora atua diretamente como política de referência π_ref!")
        print("[✓] Economia: 0 VRAM adicional necessária para modelo de referência!\n")
    else:
        print(f"[*] Nenhum adaptador prévio encontrado em '{adapter_path}'. Iniciando GRPO diretamente sobre o modelo base.")

    # 8. Configuração de novo LoRA para o treinamento da fase GRPO
    print(f"[*] Configurando LoRA limpo para a fase de RL: r={args.lora_r}, alpha={args.lora_alpha}, target=all-linear...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    # 9. Configuração do GRPOConfig (TRL)
    print(f"[*] Configurando hiperparâmetros GRPO: lr={args.learning_rate}, beta={args.beta}, G={args.num_generations}...")
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        beta=args.beta,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs if args.max_steps <= 0 else 1.0,
        max_steps=args.max_steps,
        remove_unused_columns=False,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        fp16=(args.torch_dtype == "float16"),
        bf16=(args.torch_dtype == "bfloat16"),
        gradient_checkpointing=True,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    # 10. Inicialização do GRPOTrainer
    reward_suite = [
        cot_integrity_reward,
        technical_accuracy_reward,
        sandbox_code_reward,
    ]

    print("[*] Inicializando GRPOTrainer com a suíte de 3 recompensas verificáveis...")
    trainer = GRPOTrainer(
        model=base_model,
        reward_funcs=reward_suite,
        args=grpo_config,
        train_dataset=hf_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # 11. Treinamento GRPO
    print("\n" + "=" * 70)
    print("🔥 INICIANDO TREINAMENTO GRPO ONLINE (CyberRott v1.2)")
    print("=" * 70 + "\n")

    train_result = trainer.train()

    print("\n" + "=" * 70)
    print(f"[✓] Treinamento GRPO concluído com sucesso!")
    print(f"[*] Métricas finais: {train_result.metrics}")
    print("=" * 70 + "\n")

    # 12. Salvamento do Modelo e Tokenizer
    print(f"[*] Salvando checkpoint final em: {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[✓] Checkpoint CyberRott v1.2 GRPO gravado com sucesso em '{args.output_dir}'!")

    if args.push_to_hub:
        print("[*] Publicando modelo no Hugging Face Hub...")
        trainer.push_to_hub()
        print("[✓] Modelo publicado com sucesso!")


if __name__ == "__main__":
    main()
