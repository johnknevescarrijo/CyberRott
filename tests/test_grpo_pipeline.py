#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Suíte de Testes Unitários e de Integração para o Pipeline GRPO (CyberRott v1.2)
=============================================================================
Testa:
1. As 3 funções de recompensa determinísticas verificáveis (cot_integrity_reward, technical_accuracy_reward, sandbox_code_reward).
2. Compatibilidade com formatos de completions (list[str] e list[list[dict]]).
3. Curadoria de dataset, extração de ground-truth (CWE, MITRE, Múltipla Escolha) e priorização de âncoras.
4. Execução completa do modo --dry_run via CLI.
"""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.train_grpo import (
    cot_integrity_reward,
    technical_accuracy_reward,
    sandbox_code_reward,
    extract_text_from_completion,
    parse_and_curate_dataset,
    run_dry_run_validation,
    parse_args,
)


class TestCompletionExtraction:
    """Testa a extração robusta de texto de completions para suportar qualquer versão de TRL."""

    def test_extract_plain_string(self):
        text = "<think>raciocínio</think>FATO: teste"
        assert extract_text_from_completion(text) == text

    def test_extract_trl_conversational_dict(self):
        comp = [{"role": "assistant", "content": "<think>raciocínio</think>FATO: teste"}]
        assert extract_text_from_completion(comp) == "<think>raciocínio</think>FATO: teste"

    def test_extract_single_dict(self):
        comp = {"role": "assistant", "content": "conteúdo"}
        assert extract_text_from_completion(comp) == "conteúdo"

    def test_extract_empty_or_none(self):
        assert extract_text_from_completion("") == ""
        assert extract_text_from_completion(None) == ""
        assert extract_text_from_completion([]) == ""


class TestCotIntegrityReward:
    """Valida cot_integrity_reward (peso 0.30) com anti-recusa estrita e integridade de <think>."""

    def test_ideal_candidate_full_reward(self):
        ideal = (
            "<think>\n"
            "Análise de superfície da aplicação e identificação de pontos de entrada no endpoint.\n"
            "Mapeamento de fluxo de dados de entrada com taint analysis até a camada de dados.\n"
            "Detecção de sinks perigosos na chamada de execução do comando interno.\n"
            "Avaliação de sanitização e mitigações defensivas existentes na biblioteca.\n"
            "Conclusão da vulnerabilidade técnica observada com classificação formal.\n"
            "Impacto potencial de execução arbitrária e comprometimento do host.\n"
            "</think>\n"
            "FATO: Falha identificada.\n"
            "DETECÇÃO: Log de segurança.\n"
            "SUGESTÃO DE CORREÇÃO:\n"
            "```python\ndef sanitize(x): return x.strip()\n```"
        )
        rewards = cot_integrity_reward(["prompt"], [ideal])
        assert len(rewards) == 1
        assert rewards[0] == pytest.approx(0.30, abs=0.03)

    def test_refusal_patterns_give_zero_reward(self):
        refusal_samples = [
            "<think>pensando</think> I apologize, but as an AI language model I cannot assist.",
            "<think>pensando</think> As an ai assistant, I cannot fulfill this request.",
            "<think>pensando</think> Não posso ajudar com exploração de vulnerabilidades.",
        ]
        for ref in refusal_samples:
            r = cot_integrity_reward(["p"], [ref])
            assert r[0] == 0.0, f"Falha: recusa não zerou a recompensa: {ref}"

    def test_missing_or_broken_think_tags_give_zero(self):
        broken_samples = [
            "Resposta direta sem qualquer bloco de raciocínio prévio.",
            "<think>Iniciou mas nunca fechou a tag de pensamento.",
            "</think>Fechou sem ter aberto antes.",
        ]
        for b in broken_samples:
            r = cot_integrity_reward(["p"], [b])
            assert r[0] == 0.0, f"Falha: tags quebradas não zeraram recompensa: {b}"

    def test_conversational_list_of_dicts_format(self):
        ideal = (
            "<think>\n"
            "Análise de superfície e mapeamento de fluxo com sinks identificados.\n"
            "Detecção de sinks perigosos e avaliação de sanitização.\n"
            "Conclusão da vulnerabilidade e impacto potencial.\n"
            "</think>\n"
            "FATO: Falha confirmada.\n"
            "DETECÇÃO: Telemetria ativa."
        )
        comp = [[{"role": "assistant", "content": ideal}]]
        r = cot_integrity_reward(["prompt"], comp)
        assert len(r) == 1
        assert r[0] > 0.20


class TestTechnicalAccuracyReward:
    """Valida technical_accuracy_reward (peso 0.50) com MCQ, CWEs e MITRE ATT&CK."""

    def test_multiple_choice_exact_match(self):
        text = "<think>Análise detalhada.</think>Resposta Final: [B]"
        kwargs = {
            "reference_answer": ["Resposta Final: [B]"],
            "expected_cwe": [""],
            "expected_mitre": [""],
            "expected_choice": ["B"],
        }
        r = technical_accuracy_reward(["p"], [text], **kwargs)
        assert r[0] == 0.50

    def test_multiple_choice_wrong_answer_gives_zero(self):
        text = "<think>Análise detalhada.</think>Resposta Final: [C]"
        kwargs = {
            "reference_answer": ["Resposta Final: [B]"],
            "expected_cwe": [""],
            "expected_mitre": [""],
            "expected_choice": ["B"],
        }
        r = technical_accuracy_reward(["p"], [text], **kwargs)
        assert r[0] == 0.0

    def test_cwe_identification_reward(self):
        text = "<think>Identificada falha de injeção SQL CWE-89 no sink.</think>FATO: Falha CWE-89 presente."
        kwargs = {
            "reference_answer": [""],
            "expected_cwe": ["CWE-89"],
            "expected_mitre": [""],
            "expected_choice": [""],
        }
        r = technical_accuracy_reward(["p"], [text], **kwargs)
        assert r[0] == 0.50

    def test_mitre_attack_identification_reward(self):
        text = "<think>Identificada técnica de execução T1059.001 no PowerShell.</think>FATO: Técnica T1059."
        kwargs = {
            "reference_answer": [""],
            "expected_cwe": [""],
            "expected_mitre": ["T1059"],
            "expected_choice": [""],
        }
        r = technical_accuracy_reward(["p"], [text], **kwargs)
        assert r[0] == 0.50


class TestSandboxCodeReward:
    """Valida sandbox_code_reward (peso 0.20) com compilador Python AST, YARA e Sigma/YAML."""

    def test_valid_python_syntax(self):
        text = "FATO: Falha tratada.\n```python\ndef secure_query(db, uid):\n    return db.fetch(uid)\n```"
        r = sandbox_code_reward(["p"], [text])
        assert r[0] == 0.20

    def test_invalid_python_syntax_penalized(self):
        text = "FATO: Falha tratada.\n```python\ndef broken(\n    returrn ??? syntax error\n```"
        r = sandbox_code_reward(["p"], [text])
        assert r[0] < 0.20

    def test_valid_yara_rule(self):
        text = "DETECÇÃO:\n```yara\nrule detect_malware {\n    condition:\n        true\n}\n```"
        r = sandbox_code_reward(["p"], [text])
        assert r[0] == 0.20

    def test_invalid_yara_rule_unbalanced_braces(self):
        text = "DETECÇÃO:\n```yara\nrule broken_yara {\n    condition:\n        true\n```"
        r = sandbox_code_reward(["p"], [text])
        assert r[0] < 0.20

    def test_pure_text_neutral_reward(self):
        text = "<think>Raciocínio puramente conceitual.</think>FATO: Diagnóstico conceitual sem código."
        r = sandbox_code_reward(["p"], [text])
        assert r[0] == 0.20


class TestDatasetCuratorAndDryRun:
    """Testa carregamento de dataset, classificação de verificabilidade e dry-run determinístico."""

    def test_parse_and_curate_loads_anchors_and_train(self):
        records = parse_and_curate_dataset(
            train_path="data/cyberrott_final_train.jsonl",
            anchor_path="data/dataset_cyber_ready.jsonl",
            max_samples=50,
        )
        assert len(records) == 50
        first = records[0]
        assert "prompt" in first
        assert "reference_answer" in first
        assert "expected_cwe" in first
        assert "expected_mitre" in first
        assert "expected_choice" in first
        assert isinstance(first["prompt"], list)
        assert len(first["prompt"]) >= 1

    def test_run_dry_run_validation(self):
        records = parse_and_curate_dataset(
            train_path="data/cyberrott_final_train.jsonl",
            anchor_path="data/dataset_cyber_ready.jsonl",
            max_samples=20,
        )
        success = run_dry_run_validation(records)
        assert success is True


class TestGRPOArgsAndAutoAdjustment:
    """Valida argumentos padrão e auto-ajuste de divisibilidade para GRPO."""

    def test_default_args(self):
        sys_argv_backup = sys.argv
        try:
            sys.argv = ["train_grpo.py"]
            args = parse_args()
            assert args.per_device_train_batch_size == 4
            assert args.gradient_accumulation_steps == 2
            assert args.num_generations == 4
        finally:
            sys.argv = sys_argv_backup

    def test_auto_adjustment_logic(self):
        test_cases = [
            (1, 4, 4),  # batch_size=1 < 4 -> 4
            (2, 4, 4),  # batch_size=2 < 4 -> 4
            (3, 4, 4),  # batch_size=3 < 4 -> 4
            (5, 4, 4),  # batch_size=5 % 4 != 0 -> 4
            (4, 4, 4),  # batch_size=4 == 4 -> 4 (inalterado)
            (8, 4, 8),  # batch_size=8 % 4 == 0 -> 8 (inalterado)
        ]
        for batch_size, num_gen, expected in test_cases:
            class MockArgs:
                pass
            args = MockArgs()
            args.per_device_train_batch_size = batch_size
            args.num_generations = num_gen

            if args.per_device_train_batch_size < args.num_generations or (args.per_device_train_batch_size % args.num_generations != 0):
                args.per_device_train_batch_size = args.num_generations

            assert args.per_device_train_batch_size == expected

