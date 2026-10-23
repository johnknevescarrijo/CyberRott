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

import inspect
from scripts.train_grpo import (
    cot_integrity_reward,
    technical_accuracy_reward,
    sandbox_code_reward,
    extract_text_from_completion,
    parse_and_curate_dataset,
    run_dry_run_validation,
    parse_args,
    apply_grpo_compatibility_patches,
    enable_fast_generation_with_kv_cache,
    resolve_resume_checkpoint,
    get_eos_token_ids,
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
            assert args.gradient_accumulation_steps == 1
            assert args.num_generations == 4
            assert args.max_prompt_length == 768
            assert args.max_completion_length == 512
            assert args.max_samples == 80
            assert args.save_steps == 20
            assert args.resume_from_checkpoint is None
            assert args.gradient_checkpointing is None
            assert args.use_vllm is False
            assert args.vllm_gpu_memory_utilization == 0.4
        finally:
            sys.argv = sys_argv_backup

    def test_eos_token_ids_extraction(self):
        """Valida que get_eos_token_ids coleta nativos e alvos ChatML/Phi-4."""
        class MockTokenizer:
            def __init__(self):
                self.eos_token_id = 100257
                self.unk_token_id = 0
                self._map = {
                    "<|end|>": 100257,
                    "<|im_end|>": 100260,
                    "<|endoftext|>": 100257,
                    "<|eot_id|>": 100262,
                }
            def convert_tokens_to_ids(self, token):
                return self._map.get(token, -1)

        tok = MockTokenizer()
        ids = get_eos_token_ids(tok)
        assert 100257 in ids
        assert 100260 in ids
        assert 100262 in ids
        assert len(ids) == len(set(ids))

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


class TestGRPOTrainerCompatibility:
    """Valida a compatibilidade de GRPOTrainer com transformers >= 4.48 / 5.x."""

    def test_grpo_trainer_sampler_signature_patched(self):
        """Garante que _get_train_sampler aceita 'train_dataset' evitando TypeError no Trainer._get_dataloader."""
        from trl.trainer.grpo_trainer import GRPOTrainer

        # Certifica-se de que o patch foi executado
        apply_grpo_compatibility_patches()

        sig_train = inspect.signature(GRPOTrainer._get_train_sampler)
        assert "train_dataset" in sig_train.parameters, (
            f"_get_train_sampler deve aceitar 'train_dataset', parâmetros atuais: {list(sig_train.parameters.keys())}"
        )

        sig_eval = inspect.signature(GRPOTrainer._get_eval_sampler)
        assert "eval_dataset" in sig_eval.parameters, (
            f"_get_eval_sampler deve aceitar 'eval_dataset', parâmetros atuais: {list(sig_eval.parameters.keys())}"
        )

    def test_grpo_trainer_dataloader_and_train_step_cpu(self, tmp_path):
        """Instancia GRPOTrainer, obtém get_train_dataloader e executa 1 step em CPU com tiny model."""
        import torch
        from datasets import Dataset
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer

        apply_grpo_compatibility_patches()

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        config = AutoConfig.for_model(
            "gpt2",
            n_layer=2,
            n_head=2,
            n_embd=64,
            vocab_size=len(tokenizer),
        )
        model = AutoModelForCausalLM.from_config(config)

        dataset = Dataset.from_dict({
            "prompt": [
                "Identifique a vulnerabilidade técnica na função vulnerable_auth().",
                "Analise o vetor de ataque no endpoint /api/v1/user.",
            ]
        })

        def dummy_accuracy_reward(prompts, completions, **kwargs):
            return [1.0] * len(prompts)

        out_dir = str(tmp_path / "grpo_test_run")
        training_args = GRPOConfig(
            output_dir=out_dir,
            learning_rate=1e-4,
            max_steps=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            num_generations=2,
            max_prompt_length=32,
            max_completion_length=16,
            report_to=[],
            use_vllm=False,
        )

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[dummy_accuracy_reward],
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,
            processing_class=tokenizer,
        )

        # 1. Valida que get_train_dataloader() executa sem TypeError
        train_dataloader = trainer.get_train_dataloader()
        assert train_dataloader is not None
        assert len(train_dataloader) > 0

        # 2. Valida que get_eval_dataloader() executa sem erro
        eval_dataloader = trainer.get_eval_dataloader()
        assert eval_dataloader is not None

        # 3. Executa 1 step real de treinamento em CPU
        train_result = trainer.train()
        assert train_result is not None
        assert train_result.global_step == 1
        assert "train_loss" in train_result.metrics


class TestFastGenerationWithKVCache:
    """Valida a aceleração de rollouts reativando KV-cache com gradient checkpointing ativo."""

    def test_kv_cache_active_during_generate_and_gc_restored_after(self):
        """Verifica que durante model.generate() o use_cache permanece True e gradient_checkpointing fica False,
        e que logo após model.generate() terminar o gradient_checkpointing volta a ser True e use_cache False."""
        import torch

        class MockDecoderLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gradient_checkpointing = True

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = MockDecoderLayer()
                self.is_gradient_checkpointing = True
                self.config = type("Config", (), {"use_cache": False})()
                self.generation_config = type("GenConfig", (), {"use_cache": False})()
                self.recorded_state_during_generate = {}

            def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
                self.is_gradient_checkpointing = True
                self.layer.gradient_checkpointing = True

            def gradient_checkpointing_disable(self):
                self.is_gradient_checkpointing = False
                self.layer.gradient_checkpointing = False

            def generate(self, *args, **kwargs):
                self.recorded_state_during_generate = {
                    "kwargs_use_cache": kwargs.get("use_cache"),
                    "config_use_cache": getattr(self.config, "use_cache", None),
                    "gen_config_use_cache": getattr(self.generation_config, "use_cache", None),
                    "model_is_gc": getattr(self, "is_gradient_checkpointing", None),
                    "layer_gc": getattr(self.layer, "gradient_checkpointing", None),
                    "is_training": self.training,
                }
                return torch.tensor([[101, 102, 103]])

        model = MockModel()
        model.train()

        # Estado inicial pré-wrapper
        assert model.training is True
        assert model.is_gradient_checkpointing is True
        assert model.layer.gradient_checkpointing is True
        assert model.config.use_cache is False

        # Aplica o wrapper de aceleração KV-cache
        enable_fast_generation_with_kv_cache(model)

        # Dispara geração com generation_config explicitamente em kwargs
        custom_gen_cfg = type("CustomGenConfig", (), {"use_cache": False})()
        dummy_input = torch.tensor([[1, 2]])
        output = model.generate(input_ids=dummy_input, generation_config=custom_gen_cfg)

        # 1. Validações DURANTE model.generate():
        during = model.recorded_state_during_generate
        assert during["kwargs_use_cache"] is True, "use_cache deve ser True nos kwargs durante generate"
        assert custom_gen_cfg.use_cache is True, "custom_gen_cfg.use_cache em kwargs deve ser setado como True"
        assert during["config_use_cache"] is True, "use_cache deve ser True no model.config durante generate"
        assert during["gen_config_use_cache"] is True, "use_cache deve ser True no generation_config durante generate"
        assert during["model_is_gc"] is False, "is_gradient_checkpointing deve ser False durante generate"
        assert during["layer_gc"] is False, "layer.gradient_checkpointing deve ser False durante generate"
        assert during["is_training"] is False, "modelo deve estar em eval() durante generate para evitar descarte de cache"

        # 2. Validações APÓS model.generate():
        assert model.is_gradient_checkpointing is True, "is_gradient_checkpointing deve ser restaurado para True pós generate"
        assert model.layer.gradient_checkpointing is True, "layer.gradient_checkpointing deve ser restaurado para True pós generate"
        assert model.config.use_cache is False, "model.config.use_cache deve ser restaurado para False pós generate"
        assert model.generation_config.use_cache is False, "generation_config.use_cache deve ser restaurado para False pós generate"
        assert model.training is True, "modo de treinamento deve ser restaurado para True pós generate"

    def test_fast_generation_restores_state_even_on_exception(self):
        """Garante que mesmo se a geração falhar/interromper, o estado de treino e gradient checkpointing é restaurado."""
        import torch

        class FailingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.is_gradient_checkpointing = True
                self.config = type("Config", (), {"use_cache": False})()

            def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
                self.is_gradient_checkpointing = True

            def gradient_checkpointing_disable(self):
                self.is_gradient_checkpointing = False

            def generate(self, *args, **kwargs):
                raise RuntimeError("Simulação de OOM ou interrupção na geração")

        model = FailingModel()
        model.train()
        enable_fast_generation_with_kv_cache(model)

        with pytest.raises(RuntimeError, match="Simulação de OOM"):
            model.generate()

        assert model.is_gradient_checkpointing is True
        assert model.config.use_cache is False
        assert model.training is True

    def test_resolve_resume_checkpoint(self, tmp_path):
        """Testa a autodetecção de checkpoints ('auto') e caminhos explícitos."""
        # 1. Sem argumento -> None
        assert resolve_resume_checkpoint(None, str(tmp_path)) is None
        assert resolve_resume_checkpoint("", str(tmp_path)) is None

        # 2. 'auto' com diretório vazio ou inexistente -> None
        assert resolve_resume_checkpoint("auto", str(tmp_path / "nao_existe")) is None
        empty_dir = tmp_path / "output_empty"
        empty_dir.mkdir()
        assert resolve_resume_checkpoint("auto", str(empty_dir)) is None

        # 3. 'auto' com múltiplos checkpoints -> seleciona o mais recente por step
        run_dir = tmp_path / "output_run"
        run_dir.mkdir()
        ckpt25 = run_dir / "checkpoint-25"
        ckpt50 = run_dir / "checkpoint-50"
        ckpt100 = run_dir / "checkpoint-100"
        ckpt25.mkdir()
        ckpt50.mkdir()
        ckpt100.mkdir()

        resolved = resolve_resume_checkpoint("auto", str(run_dir))
        assert resolved == str(ckpt100)

        # 4. Caminho explícito existente
        assert resolve_resume_checkpoint(str(ckpt50), str(run_dir)) == str(ckpt50)

        # 5. Caminho explícito inexistente -> fallback None
        assert resolve_resume_checkpoint(str(run_dir / "checkpoint-999"), str(run_dir)) is None



