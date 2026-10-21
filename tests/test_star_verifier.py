#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testes unitários e de validação para scripts/star_verifier.py
"""

import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.star_verifier import (
    STaRVerifier,
    verify_format_integrity,
    verify_ground_truth,
    verify_code_syntax,
    extract_code_blocks,
    extract_mc_answer,
)

def test_refusal_detection():
    verifier = STaRVerifier()
    cand = (
        "<think>Análise de Superfície: Investigando a vulnerabilidade em questão.</think>\n"
        "As an AI language model, I cannot provide this code."
    )
    res = verifier.verify_candidate({}, cand)
    assert not res.passed, "Candidate com recusa de IA não deve passar"
    assert not res.format_ok, "format_ok deve ser False quando há recusa"
    assert any("Recusa conversacional" in r for r in res.reasons)
    print("✅ test_refusal_detection passou com sucesso.")

def test_think_format_integrity():
    verifier = STaRVerifier(min_think_chars=50, min_think_words=10)

    # 1. Sem think
    res1 = verifier.verify_candidate({}, "Resposta direta sem bloco de reflexão.")
    assert not res1.passed
    assert not res1.format_ok

    # 2. Think não fechado
    res2 = verifier.verify_candidate({}, "<think>Pensamento sem tag de fechamento...")
    assert not res2.passed
    assert not res2.format_ok

    # 3. Think muito curto
    res3 = verifier.verify_candidate({}, "<think>Curto</think>\nResposta final suficiente aqui.")
    assert not res3.passed
    assert not res3.format_ok

    # 4. Válido com estágios
    cand_valid = (
        "<think>\n"
        "Análise de Superfície: examinando a função get_user.\n"
        "Pontos de Entrada: parâmetro username controlado pelo atacante.\n"
        "Mapeamento de Fluxo: parâmetro interpolado diretamente na query SQL.\n"
        "Detecção de Sinks: cursor.execute com string formatada.\n"
        "Avaliação de Sanitização: nenhuma sanitização presente.\n"
        "Conclusão da Vulnerabilidade: SQL Injection (CWE-89).\n"
        "Impacto Potencial: bypass de autenticação e vazamento de dados.\n"
        "</think>\n\n"
        "FATO: A função possui SQL Injection.\n"
        "SUGESTÃO DE CORREÇÃO: Usar queries parametrizadas."
    )
    res4 = verifier.verify_candidate({"expected_cwe": "CWE-89"}, cand_valid)
    assert res4.passed, f"Candidato válido deveria passar, erros: {res4.reasons}"
    assert res4.format_ok
    assert res4.ground_truth_ok
    assert res4.code_syntax_ok
    assert res4.score >= 0.8
    print("✅ test_think_format_integrity passou com sucesso.")

def test_ground_truth_matching():
    verifier = STaRVerifier(min_think_chars=30, min_think_words=5)

    prob_mc = {"answer": "C"}
    cand_c = "<think>Análise técnica detalhada da questão de segurança.</think>\nResposta Final: [C]"
    cand_b = "<think>Análise técnica detalhada da questão de segurança.</think>\nResposta Final: [B]"

    res_c = verifier.verify_candidate(prob_mc, cand_c)
    res_b = verifier.verify_candidate(prob_mc, cand_b)
    assert res_c.ground_truth_ok
    assert not res_b.ground_truth_ok

    # Teste de CWE
    prob_cwe = {"expected_cwe": "CWE-79"}
    cand_xss = "<think>Análise de fluxo e sink de injeção de script.</think>\nFATO: Falha XSS mapeada como CWE-79."
    cand_sqli = "<think>Análise de fluxo de injeção de query.</think>\nFATO: Falha SQLi mapeada como CWE-89."

    assert verifier.verify_candidate(prob_cwe, cand_xss).ground_truth_ok
    assert not verifier.verify_candidate(prob_cwe, cand_sqli).ground_truth_ok
    print("✅ test_ground_truth_matching passou com sucesso.")

def test_code_syntax_sandbox():
    verifier = STaRVerifier(min_think_chars=30, min_think_words=5)

    # 1. Python com erro de sintaxe
    cand_py_broken = (
        "<think>Análise técnica detalhada do sink e correção necessária.</think>\n"
        "FATO: Problema de sintaxe no exploit.\n"
        "```python\n"
        "def broken_code(\n"
        "    if True:\n"
        "        pass\n"
        "```"
    )
    res_broken = verifier.verify_candidate({}, cand_py_broken)
    assert not res_broken.code_syntax_ok, "Código Python inválido deve falhar na sintaxe"
    assert not res_broken.passed

    # 2. Python válido
    cand_py_valid = (
        "<think>Análise técnica detalhada do sink e correção necessária.</think>\n"
        "FATO: Correção implementada com segurança.\n"
        "```python\n"
        "import sqlite3\n"
        "def query_safe(conn, user):\n"
        "    cursor = conn.cursor()\n"
        "    cursor.execute('SELECT * FROM users WHERE name = ?', (user,))\n"
        "    return cursor.fetchall()\n"
        "```"
    )
    res_py = verifier.verify_candidate({}, cand_py_valid)
    assert res_py.code_syntax_ok, f"Código Python válido falhou: {res_py.reasons}"

    # 3. YARA inválida (chaves desbalanceadas)
    cand_yara_broken = (
        "<think>Análise técnica detalhada da assinatura defensiva.</think>\n"
        "FATO: Detecção com YARA.\n"
        "```yara\n"
        "rule BrokenRule {\n"
        "    strings: $a = \"malware\"\n"
        "    condition: $a\n"
        "```"
    )
    res_yara_broken = verifier.verify_candidate({}, cand_yara_broken)
    assert not res_yara_broken.code_syntax_ok

    # 4. YARA válida
    cand_yara_valid = (
        "<think>Análise técnica detalhada da assinatura defensiva.</think>\n"
        "FATO: Detecção com YARA.\n"
        "```yara\n"
        "rule ValidRule {\n"
        "    strings:\n"
        "        $a = \"evil_payload\"\n"
        "    condition:\n"
        "        $a\n"
        "}\n"
        "```"
    )
    res_yara_valid = verifier.verify_candidate({}, cand_yara_valid)
    assert res_yara_valid.code_syntax_ok, f"YARA válida falhou: {res_yara_valid.reasons}"

    # 5. Sigma / YAML válido
    cand_sigma_valid = (
        "<think>Análise técnica detalhada para SIEM / Sigma rule.</think>\n"
        "DETECÇÃO / TELEMETRIA:\n"
        "```yaml\n"
        "title: Suspicious PowerShell Download\n"
        "logsource:\n"
        "    category: process_creation\n"
        "    product: windows\n"
        "detection:\n"
        "    selection:\n"
        "        CommandLine|contains: 'Invoke-WebRequest'\n"
        "    condition: selection\n"
        "```"
    )
    res_sigma = verifier.verify_candidate({}, cand_sigma_valid)
    assert res_sigma.code_syntax_ok, f"Sigma válida falhou: {res_sigma.reasons}"
    print("✅ test_code_syntax_sandbox passou com sucesso.")

def test_rejection_and_ranking():
    verifier = STaRVerifier(min_think_chars=30, min_think_words=5)
    problem = {"answer": "A", "expected_cwe": "CWE-89"}

    cand_bad_refusal = "<think>Pensando...</think>\nAs an AI, I cannot help."
    cand_bad_answer = "<think>Análise técnica...</think>\nResposta Final: B"
    cand_good_partial = (
        "<think>Análise de Superfície da vulnerabilidade...</think>\n"
        "Resposta Final: A"
    )
    cand_good_full = (
        "<think>\n"
        "Análise de Superfície e Pontos de Entrada.\n"
        "Mapeamento de Fluxo e Detecção de Sinks.\n"
        "Avaliação de Sanitização e Conclusão da Vulnerabilidade.\n"
        "Impacto Potencial e Mitigações.\n"
        "</think>\n\n"
        "FATO: Vulnerabilidade SQL Injection (CWE-89).\n"
        "Resposta Final: A\n"
        "SUGESTÃO DE CORREÇÃO:\n"
        "```python\n"
        "safe_query = True\n"
        "```"
    )

    candidates = [cand_bad_refusal, cand_bad_answer, cand_good_partial, cand_good_full]
    ranked = verifier.filter_and_rank_candidates(problem, candidates)

    assert len(ranked) >= 1, "Ao menos os candidatos válidos devem passar"
    # O candidato completo deve ter pontuação maior que o parcial
    top_cand, top_res = ranked[0]
    assert top_cand == cand_good_full, "O candidato mais completo deve ficar no topo do ranking"
    assert top_res.score > 0.8
    print("✅ test_rejection_and_ranking passou com sucesso.")

if __name__ == "__main__":
    test_refusal_detection()
    test_think_format_integrity()
    test_ground_truth_matching()
    test_code_syntax_sandbox()
    test_rejection_and_ranking()
    print("\n🎉 TODOS OS TESTES UNITÁRIOS DO STAR_VERIFIER FORAM APROVADOS!")
