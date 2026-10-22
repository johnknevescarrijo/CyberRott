#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testes Unitários para o Pipeline STaR v1.1 e Integração de Treinamento
Valida:
1. Prevenção de OOM em RolloutGenerator (unload/close)
2. Fallback de dados para diretório data/ em scripts/data_prep.py e scripts/train_star.py
3. Execução ponta a ponta do pipeline STaR (dry-run com acúmulo de checkpoints)
4. Integridade estrutural do Jupyter Notebook CyberRott_Colab_Training.ipynb
"""

import os
import sys
import json
import shutil
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.train_star import (
    RolloutGenerator,
    load_problem_suite,
    load_gold_anchors,
    run_star_pipeline,
    parse_args,
)
from scripts.data_prep import prepare_dataset, load_anchor_samples


class TestRolloutGeneratorOOMPrevention:
    """Valida métodos de prevenção de OOM na GPU L4/A100."""

    def test_rollout_generator_mock_initialization(self):
        gen = RolloutGenerator(
            backend="mock",
            model_name_or_path="microsoft/Phi-4-mini-instruct",
        )
        assert gen.backend == "mock"
        assert hasattr(gen, "unload")
        assert hasattr(gen, "close")
        assert hasattr(gen, "generate_candidates")

    def test_rollout_generator_unload_and_close_clears_memory(self):
        gen = RolloutGenerator(
            backend="mock",
            model_name_or_path="microsoft/Phi-4-mini-instruct"
        )
        gen.model = "dummy_model"
        gen._hf_model = "dummy_model"
        gen.tokenizer = "dummy_tokenizer"
        gen._hf_tokenizer = "dummy_tokenizer"
        gen.vllm_engine = "dummy_engine"
        gen._vllm_engine = "dummy_engine"

        gen.unload()
        assert gen.model is None
        assert gen._hf_model is None
        assert gen.tokenizer is None
        assert gen._hf_tokenizer is None
        assert gen.vllm_engine is None
        assert gen._vllm_engine is None

    def test_generate_candidates_alias(self):
        gen = RolloutGenerator(
            backend="mock",
            model_name_or_path="microsoft/Phi-4-mini-instruct"
        )
        problems = [{
            "system_prompt": "sys",
            "user_prompt": "Audit this C code with strcpy.",
            "metadata": {"expected_cwe": "CWE-120"}
        }]
        candidates = gen.generate_candidates(problems, num_candidates=3)
        assert len(candidates) == 1
        assert len(candidates[0]) == 3
        assert "<think>" in candidates[0][0]
        assert "</think>" in candidates[0][0]


class TestDataFallbackAndPreparation:
    """Valida o fallback automático de data/ e criação de diretórios."""

    def test_load_problem_suite_fallback(self):
        # Passa caminho sem prefixo 'data/'
        problems = load_problem_suite("dataset_cyber_ready.jsonl", max_problems=2)
        assert len(problems) == 2
        assert "user_prompt" in problems[0]

    def test_load_gold_anchors_fallback(self):
        anchors = load_gold_anchors("dataset_cyber_ready.jsonl", upsample_factor=1)
        assert len(anchors) > 0

    def test_prepare_dataset_fallback_and_mkdir(self):
        temp_dir = tempfile.mkdtemp()
        output_file = os.path.join(temp_dir, "nested", "subfolder", "train.jsonl")
        try:
            # Deve criar 'nested/subfolder/' automaticamente e achar 'dataset_cyber_ready.jsonl' via fallback
            result = prepare_dataset(
                anchor_file="dataset_cyber_ready.jsonl",
                anchor_repeat=1,
                preset="v1.0",
                max_hf_samples=0,  # apenas âncoras locais
                output_file=output_file
            )
            assert os.path.exists(output_file)
            assert len(result) > 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestStarPipelineDryRun:
    """Valida execução completa ponta a ponta com acúmulo contínuo de checkpoints."""

    def test_run_star_pipeline_dry_run_multi_iteration(self):
        temp_out = tempfile.mkdtemp()
        try:
            args = parse_args([])
            args.dry_run = True
            args.max_problems = 2
            args.star_iterations = 2
            args.num_candidates = 4
            args.problem_file = "dataset_cyber_ready.jsonl"
            args.anchor_file = "dataset_cyber_ready.jsonl"
            args.output_dir = temp_out

            final_adapter = run_star_pipeline(args)
            assert final_adapter is not None
            assert os.path.exists(final_adapter)
            assert os.path.exists(os.path.join(final_adapter, "adapter_config.json"))

            # Valida telemetria e resumo consolidado
            summary_path = os.path.join(temp_out, "star_pipeline_summary.json")
            assert os.path.exists(summary_path)
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            assert summary["total_iterations"] == 2
            assert len(summary["history"]) == 2
        finally:
            shutil.rmtree(temp_out, ignore_errors=True)


class TestColabNotebookIntegrity:
    """Valida se o notebook Jupyter gerado é válido e contém as seções necessárias."""

    def test_notebook_valid_json_and_v4(self):
        nb_path = "notebooks/CyberRott_Colab_Training.ipynb"
        assert os.path.exists(nb_path), f"Notebook não encontrado em {nb_path}"

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        assert nb.get("nbformat") == 4
        assert "cells" in nb
        assert len(nb["cells"]) >= 7

        all_text = " ".join("".join(c.get("source", [])) for c in nb["cells"])
        # Valida menção aos 3 estágios e GPUs alvo
        assert "CyberRott v1.1" in all_text
        assert "L4" in all_text
        assert "data_prep.py" in all_text
        assert "train_colab.py" in all_text
        assert "train_star.py" in all_text
        assert "STaRVerifier" in all_text or "STaR" in all_text
