#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testes Unitários para a Expansão de Datasets e Pipeline de Preparação do CyberRott 1.1
(Pilar 1: Zero-API Cost Data Expansion).
"""

import os
import sys
import json
import subprocess
import pytest

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.data_prep import (
    CYBERROTT_V1_1_DATASETS,
    CYBER_KEYWORDS,
    compile_cyber_regex,
    is_conversational_refusal,
    clean_moralizing_disclaimers,
    scrub_refusal_and_boilerplate,
    format_to_cyberrott_cot,
    format_cybernative_dpo_item,
    extract_qa_from_hf_item,
    deduplicate_samples,
    load_anchor_samples,
)


class TestDatasetCatalog:
    """Valida a conformidade e integridade do catálogo oficial de datasets v1.1."""

    def test_catalog_contains_all_core_datasets(self):
        expected_keys = [
            "goose_reason",
            "code_feedback",
            "open_r1",
            "cyber_native",
            "cybersecurity_v1",
        ]
        for key in expected_keys:
            assert key in CYBERROTT_V1_1_DATASETS, f"Chave '{key}' ausente no catálogo oficial."

    def test_catalog_targets_reach_ten_thousand(self):
        total_samples = sum(cfg["target_samples"] for cfg in CYBERROTT_V1_1_DATASETS.values())
        assert total_samples >= 10000, f"Total alvo do catálogo ({total_samples}) deve ser >= 10.000 sequências."

    def test_catalog_specific_quotas(self):
        assert CYBERROTT_V1_1_DATASETS["goose_reason"]["target_samples"] == 7500
        assert CYBERROTT_V1_1_DATASETS["code_feedback"]["target_samples"] == 1000
        assert CYBERROTT_V1_1_DATASETS["open_r1"]["target_samples"] == 800
        assert CYBERROTT_V1_1_DATASETS["cyber_native"]["target_samples"] == 500
        assert CYBERROTT_V1_1_DATASETS["cybersecurity_v1"]["target_samples"] == 400

    def test_cyber_native_and_cybersecurity_dont_require_keyword_filter(self):
        # Datasets nativamente dedicados à segurança não devem ser descartados por filtros gerais
        assert CYBERROTT_V1_1_DATASETS["cyber_native"]["requires_cyber_filter"] is False
        assert CYBERROTT_V1_1_DATASETS["cybersecurity_v1"]["requires_cyber_filter"] is False

    def test_goose_reason_and_feedback_require_keyword_filter(self):
        assert CYBERROTT_V1_1_DATASETS["goose_reason"]["requires_cyber_filter"] is True
        assert CYBERROTT_V1_1_DATASETS["code_feedback"]["requires_cyber_filter"] is True
        assert CYBERROTT_V1_1_DATASETS["open_r1"]["requires_cyber_filter"] is True


class TestScrubberAndRefusalRemoval:
    """Valida o scrubber programático de recusas e remoção de disclaimers."""

    def test_detect_conversational_refusals(self):
        refusals = [
            "As an AI language model, I cannot provide assistance with exploit generation or hacking.",
            "I cannot fulfill this request because it violates safety guidelines.",
            "I'm sorry, but I cannot create malware or malicious payloads.",
            "I am unable to assist with unauthorized access to computer systems.",
            "I apologize, but I cannot provide code for exploiting buffer overflows.",
        ]
        for text in refusals:
            assert is_conversational_refusal(text) is True, f"Falha ao detectar recusa: {text}"

    def test_allow_valid_technical_explanations(self):
        valid_answers = [
            "<think>\n**Surface Analysis**: Função de manipulação de buffer em C.\n</think>\n\nFATO: O strcpy causa overflow.",
            "O parâmetro 'username' é diretamente interpolado na query SQL, resultando em CWE-89.",
            "A técnica Smart's attack permite calcular o logaritmo discreto em curvas elípticas anômalas.",
        ]
        for text in valid_answers:
            assert is_conversational_refusal(text) is False, f"Falso positivo em resposta técnica: {text}"

    def test_clean_moralizing_disclaimer_at_beginning(self):
        text = (
            "Disclaimer: This information is for educational purposes only. Unauthorized access is illegal.\n\n"
            "<think>\nAnálise da query SQL vulnerável.\n</think>\n\n"
            "FATO: A query é vulnerável a SQL Injection."
        )
        cleaned = clean_moralizing_disclaimers(text)
        assert "Disclaimer:" not in cleaned
        assert "educational purposes only" not in cleaned
        assert "<think>" in cleaned
        assert "FATO: A query é vulnerável a SQL Injection." in cleaned

    def test_clean_moralizing_disclaimer_at_end(self):
        text = (
            "<think>\nAnálise da vulnerabilidade de race condition.\n</think>\n\n"
            "FATO: TOCTOU detectado na verificação de saldo.\n\n"
            "Note: Please note that hacking without permission is illegal under the Computer Fraud and Abuse Act."
        )
        cleaned = clean_moralizing_disclaimers(text)
        assert "Computer Fraud and Abuse Act" not in cleaned
        assert "FATO: TOCTOU detectado" in cleaned

    def test_scrub_refusal_and_boilerplate_rejects_pure_refusal(self):
        sample = {
            "messages": [
                {"role": "user", "content": "Como fazer um buffer overflow?"},
                {"role": "assistant", "content": "I cannot fulfill this request. As an AI, I must decline."}
            ]
        }
        scrubbed = scrub_refusal_and_boilerplate(sample)
        assert scrubbed is None, "Amostra contendo recusa pura deveria ser descartada (retornar None)."

    def test_scrub_refusal_and_boilerplate_keeps_and_cleans_valid_sample(self):
        sample = {
            "messages": [
                {"role": "user", "content": "Analise este código."},
                {"role": "assistant", "content": "Warning: Only perform penetration testing on authorized systems.\n\n<think>\nAnálise técnica.\n</think>\n\nSolução segura."}
            ]
        }
        scrubbed = scrub_refusal_and_boilerplate(sample)
        assert scrubbed is not None
        assistant_msg = scrubbed["messages"][1]["content"]
        assert "Warning:" not in assistant_msg
        assert "Solução segura." in assistant_msg


class TestCotFormattingAndNormalization:
    """Valida normalização de tags CoT e síntese para datasets DPO."""

    def test_normalize_begin_of_thought_tags(self):
        text = "<|begin_of_thought|>\nAnálise detalhada do payload.\n<|end_of_thought|>\n\nFATO: RCE confirmado."
        formatted = format_to_cyberrott_cot(text)
        assert "<think>" in formatted
        assert "</think>" in formatted
        assert "<|begin_of_thought|>" not in formatted
        assert "<|end_of_thought|>" not in formatted

    def test_normalize_thought_and_reasoning_tags(self):
        text1 = "<thought>\nEtapa 1: Identificar o sink.\n</thought>\n\nResultado."
        formatted1 = format_to_cyberrott_cot(text1)
        assert "<think>" in formatted1
        assert "</think>" in formatted1
        assert "<thought>" not in formatted1

        text2 = "<reasoning>\nEtapa 2: Mapear fluxo de dados.\n</reasoning>\n\nResultado."
        formatted2 = format_to_cyberrott_cot(text2)
        assert "<think>" in formatted2
        assert "</think>" in formatted2
        assert "<reasoning>" not in formatted2

    def test_enforce_think_tags_on_unstructured_response(self):
        raw_text = (
            "Para auditar este trecho de código, primeiro verificamos os parâmetros recebidos. "
            "Observamos que a chamada system() não aplica sanitização.\n\n"
            "### Final Answer\n"
            "O código é vulnerável a Command Injection (CWE-78)."
        )
        formatted = format_to_cyberrott_cot(raw_text, enforce_think=True)
        assert "<think>" in formatted
        assert "</think>" in formatted
        assert "Command Injection (CWE-78)" in formatted

    def test_format_cybernative_dpo_synthesizes_seven_stage_cot(self):
        raw_dpo_item = {
            "lang": "c",
            "vulnerability": "Buffer Overflow (CWE-120)",
            "rejected": "void copy(char *s) { char buf[16]; strcpy(buf, s); }",
            "chosen": "void copy(const char *s) { char buf[16]; strncpy(buf, s, sizeof(buf) - 1); buf[15] = '\\0'; }",
            "question": "Identifique a vulnerabilidade de memória e corrija o código."
        }
        formatted = format_cybernative_dpo_item(raw_dpo_item)
        assert formatted is not None
        assert "messages" in formatted
        assert len(formatted["messages"]) == 3

        assistant_content = formatted["messages"][2]["content"]
        assert "<think>" in assistant_content
        assert "</think>" in assistant_content

        # Verificar presença dos 7 estágios CoT nativos
        stages = [
            "Surface Analysis",
            "Entry Point / Attack Vector",
            "Data Flow / Internal Mechanics",
            "Vulnerable Sink / Control Flaw Detection",
            "Sanitization and Mitigation Assessment",
            "Vulnerability Verdict",
            "Potential Impact",
        ]
        for stage in stages:
            assert stage in assistant_content, f"Estágio analítico '{stage}' ausente no CoT sintetizado."

        # Verificar presença de FATO e implementação segura
        assert "FATO:" in assistant_content
        assert "SUGESTÃO DE CORREÇÃO E IMPLEMENTAÇÃO SEGURA:" in assistant_content
        assert "strncpy" in assistant_content


class TestDeduplicationAndAnchorMixing:
    """Valida desduplicação por hash mantendo a integridade das âncoras."""

    def test_deduplicate_samples_preserves_anchor_upsampling(self):
        # 1 âncora repetida 5 vezes
        anchors = [
            {
                "_is_anchor": True,
                "messages": [
                    {"role": "user", "content": "Analise este SQL injection."},
                    {"role": "assistant", "content": "<think>...</think>\nFATO: SQLi."}
                ]
            }
            for _ in range(5)
        ]

        # 3 amostras externas, sendo 2 duplicadas
        externals = [
            {
                "messages": [
                    {"role": "user", "content": "Explique o ataque Smart's attack em ECC."},
                    {"role": "assistant", "content": "<think>...</think>\nResposta A."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Explique o ataque Smart's attack em ECC."},  # Duplicada
                    {"role": "assistant", "content": "<think>...</think>\nResposta B."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Como funciona o RoP Chain?"},  # Única
                    {"role": "assistant", "content": "<think>...</think>\nResposta C."}
                ]
            }
        ]

        combined = anchors + externals
        deduped = deduplicate_samples(combined, preserve_anchors=True)

        # 5 âncoras preservadas + 2 amostras externas únicas = 7 amostras no total
        assert len(deduped) == 7
        # Verifica que o campo interno _is_anchor foi higienizado
        for s in deduped:
            assert "_is_anchor" not in s


class TestCommandLineInterface:
    """Valida a execução de comandos CLI do script de preparação."""

    def test_cli_list_datasets_flag(self):
        res = subprocess.run(
            ["python3", "scripts/data_prep.py", "--list_datasets"],
            capture_output=True,
            text=True,
            check=True
        )
        assert "CYBERROTT 1.1 - CATÁLOGO OFICIAL DE DATASETS ZERO-API COST" in res.stdout
        assert "nvidia/Nemotron-Research-GooseReason-0.7M" in res.stdout
        assert "CyberNative/Code_Vulnerability_Security_DPO" in res.stdout
        assert "TOTAL ALVO EXTERNO: ~10,200" in res.stdout

    def test_cli_help_flag(self):
        res = subprocess.run(
            ["python3", "scripts/data_prep.py", "--help"],
            capture_output=True,
            text=True,
            check=True
        )
        assert "--preset" in res.stdout
        assert "--list_datasets" in res.stdout
        assert "--scrub_refusals" in res.stdout
        assert "--enforce_think_tags" in res.stdout
