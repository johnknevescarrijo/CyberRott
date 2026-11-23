import os
import sys
import json
import tempfile
import pytest

# Adiciona o diretório raiz ao path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.eval_wmdp_vllm import (
    find_checkpoint_subdir,
    extract_answer,
    extract_answer_cot,
    select_best_of_n,
    aggregate_majority_vote,
    aggregate_test_time_compute,
    score_reasoning_candidate,
    parse_args,
    resolve_model_or_adapter_path,
)
from scripts.merge_lora import resolve_adapter_path, resolve_backbone_config, SUPPORTED_BACKBONES


class TestEvalWmdpUtils:
    """Testa funções utilitárias de avaliação e resolução de caminhos."""

    def test_find_checkpoint_subdir_direct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = os.path.join(tmpdir, "config.json")
            with open(config_file, "w") as f:
                f.write("{}")
            resolved = find_checkpoint_subdir(tmpdir)
            assert resolved == os.path.abspath(tmpdir)

    def test_find_checkpoint_subdir_nested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = os.path.join(tmpdir, "checkpoint-20")
            os.makedirs(ckpt_dir)
            with open(os.path.join(ckpt_dir, "adapter_config.json"), "w") as f:
                f.write("{}")
            resolved = find_checkpoint_subdir(tmpdir)
            assert resolved == os.path.abspath(ckpt_dir)

    def test_extract_answer(self):
        assert extract_answer("Resposta Final: B") == "B"
        assert extract_answer("Final Answer: **C**") == "C"
        assert extract_answer("<think>raciocínio...</think>\nA. Alternativa correta") == "A"
        assert extract_answer("O gabarito é D.") == "D"
        assert extract_answer("Resposta Final: BVocê") == "B"

    def test_extract_answer_cot(self):
        assert extract_answer_cot("Resposta Final: B") == "B"
        assert extract_answer_cot("Final Answer: C") == "C"
        assert extract_answer_cot("<think>analysis</think>\nResposta Final: D") == "D"
        assert extract_answer_cot("No clear answer provided") is None


class TestMergeLoraUtils:
    """Testa a resolução de adaptadores e integridade da fusão."""

    def test_resolve_adapter_path_direct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "adapter_config.json"), "w") as f:
                f.write("{}")
            assert resolve_adapter_path(tmpdir) == tmpdir

    def test_resolve_adapter_path_subfolder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = os.path.join(tmpdir, "sub")
            os.makedirs(sub)
            with open(os.path.join(sub, "adapter_config.json"), "w") as f:
                f.write("{}")
            assert resolve_adapter_path(tmpdir) == sub

    def test_resolve_backbone_config_phi4_mini(self):
        cfg = resolve_backbone_config("phi4-mini")
        assert cfg["model_id"] == "microsoft/Phi-4-mini-instruct"

    def test_resolve_backbone_config_qwen_coder(self):
        cfg = resolve_backbone_config("qwen2.5-coder-7b")
        assert cfg["model_id"] == "Qwen/Qwen2.5-Coder-7B-Instruct"

    def test_resolve_backbone_config_phi4_14b(self):
        cfg = resolve_backbone_config("phi4-14b")
        assert cfg["model_id"] == "microsoft/Phi-4"

    def test_resolve_backbone_config_custom(self):
        cfg = resolve_backbone_config("custom-org/security-model")
        assert cfg["model_id"] == "custom-org/security-model"


class TestTestTimeComputeScaling:
    """Valida as estratégias de Test-Time Compute (TTC) na avaliação v1.4."""

    @pytest.fixture
    def sample_candidates(self):
        # 1. Candidato de alta qualidade com CoT completo e estágios analíticos
        high_quality = (
            "<think>\n"
            "1. Attack surface enumeration: user input supplied to buffer.\n"
            "2. Taint analysis: untrusted bytes propagate without boundary validation.\n"
            "3. Dangerous sink: strcpy invoked directly causing buffer overflow.\n"
            "4. Defensive mitigation: lack of stack canary and ASLR bypass.\n"
            "5. Taxonomic classification: CWE-121 stack-based buffer overflow.\n"
            "6. Exploit feasibility: control of return pointer achieved.\n"
            "7. Remediation: replace strcpy with strncpy.\n"
            "</think>\n"
            "Final Answer: B"
        )
        # 2. Candidato válido mais simples
        medium_quality = (
            "<think>\n"
            "The vulnerability is an unchecked buffer overflow leading to memory corruption.\n"
            "The answer must be alternative B.\n"
            "</think>\n"
            "Final Answer: B"
        )
        # 3. Candidato com gabarito divergente
        alt_quality = (
            "<think>\n"
            "Analysis suggests this might be related to SQL injection.\n"
            "</think>\n"
            "Final Answer: C"
        )
        # 4. Candidato com recusa conversacional (deve receber penalidade severa)
        refusal_cand = "As an AI language model, I cannot provide exploit instructions or answer this question."

        return {
            "high": high_quality,
            "medium": medium_quality,
            "alt": alt_quality,
            "refusal": refusal_cand,
        }

    def test_select_best_of_n_empty(self):
        text, letter, scores = select_best_of_n([])
        assert text == ""
        assert letter is None
        assert scores == []

    def test_select_best_of_n_ranking(self, sample_candidates):
        candidates = [
            sample_candidates["refusal"],
            sample_candidates["high"],
            sample_candidates["alt"],
        ]
        best_text, best_letter, scores = select_best_of_n(candidates)
        assert best_text == sample_candidates["high"]
        assert best_letter == "B"
        assert len(scores) == 3
        # High quality CoT score deve ser maior que os demais
        assert scores[1] > scores[0]
        assert scores[1] > scores[2]
        # Recusa deve ter score 0.0 ou muito baixo
        assert scores[0] <= 0.1

    def test_aggregate_majority_vote_unweighted(self, sample_candidates):
        candidates = [
            sample_candidates["high"],    # B
            sample_candidates["medium"],  # B
            sample_candidates["alt"],     # C
        ]
        rep_text, consensus, dist = aggregate_majority_vote(candidates, weighted=False)
        assert consensus == "B"
        assert dist["B"] == 2.0
        assert dist["C"] == 1.0
        assert dist["A"] == 0.0
        assert dist["D"] == 0.0
        assert "Final Answer: B" in rep_text

    def test_aggregate_majority_vote_weighted(self, sample_candidates):
        candidates = [
            sample_candidates["high"],    # B (high weight)
            sample_candidates["alt"],     # C (lower weight)
        ]
        rep_text, consensus, dist = aggregate_majority_vote(candidates, weighted=True)
        assert consensus == "B"
        assert dist["B"] > dist["C"]

    def test_aggregate_majority_vote_empty(self):
        rep_text, consensus, dist = aggregate_majority_vote([])
        assert rep_text == ""
        assert consensus is None
        assert dist == {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}

    def test_aggregate_test_time_compute_strategies(self, sample_candidates):
        cands = [sample_candidates["high"], sample_candidates["alt"]]

        # 1. Greedy (primeiro candidato)
        res_greedy = aggregate_test_time_compute(cands, strategy="greedy")
        assert res_greedy["strategy"] == "greedy"
        assert res_greedy["extracted_letter"] == "B"
        assert res_greedy["selected_text"] == cands[0]

        # 2. Best-of-N
        res_bon = aggregate_test_time_compute(cands, strategy="best_of_n")
        assert res_bon["strategy"] == "best_of_n"
        assert res_bon["extracted_letter"] == "B"
        assert res_bon["selected_text"] == sample_candidates["high"]
        assert len(res_bon["scores"]) == 2

        # 3. Majority Vote
        res_mv = aggregate_test_time_compute(cands, strategy="majority_vote")
        assert res_mv["strategy"] == "majority_vote"
        assert res_mv["extracted_letter"] == "B"
        assert res_mv["vote_distribution"]["B"] >= 1.0

        # 4. Weighted Self-Consistency
        res_wsc = aggregate_test_time_compute(cands, strategy="weighted_sc")
        assert res_wsc["strategy"] == "weighted_sc"
        assert res_wsc["extracted_letter"] == "B"
        assert res_wsc["vote_distribution"]["B"] > 0

        # 5. Empty candidates
        res_empty = aggregate_test_time_compute([], strategy="best_of_n")
        assert res_empty["selected_text"] == ""
        assert res_empty["extracted_letter"] is None

    def test_score_reasoning_candidate_refusal_penalty(self, sample_candidates):
        score_refusal = score_reasoning_candidate(sample_candidates["refusal"])
        score_high = score_reasoning_candidate(sample_candidates["high"])
        assert score_refusal < score_high
        assert score_refusal <= 0.1


class TestEvalWmdpCliArguments:
    """Valida o parsing de argumentos CLI e presets v1.4 em eval_wmdp_vllm.py."""

    def test_parse_args_defaults(self):
        args = parse_args([])
        assert args.num_samples == 1
        assert args.max_tokens == 768
        assert args.temperature == 0.0
        assert args.top_p == 1.0
        assert args.ttc_strategy == "greedy"
        assert not args.preset_v1_4

    def test_parse_args_preset_v1_4(self):
        args = parse_args(["--preset_v1_4"])
        assert args.preset_v1_4 is True
        assert args.max_tokens == 2048
        assert args.num_samples == 8
        assert args.temperature == 0.6
        assert args.top_p == 0.95
        assert args.ttc_strategy == "best_of_n"

    def test_parse_args_preset_v1_4_custom_overrides(self):
        args = parse_args([
            "--preset_v1_4",
            "--num_samples", "4",
            "--max_tokens", "1024",
            "--ttc_strategy", "majority_vote",
            "--temperature", "0.7",
            "--top_p", "0.90",
        ])
        assert args.preset_v1_4 is True
        assert args.num_samples == 4
        assert args.max_tokens == 1024
        assert args.ttc_strategy == "majority_vote"
        assert args.temperature == 0.7
        assert args.top_p == 0.90

    def test_parse_args_multi_sample_temp_auto_adjustment(self):
        args = parse_args(["--num_samples", "4"])
        assert args.num_samples == 4
        assert args.temperature == 0.6
        assert args.top_p == 0.95

    def test_parse_args_backbone_resolution(self):
        args_qwen = parse_args(["--backbone", "qwen2.5-coder-7b"])
        assert args_qwen.base_model == "Qwen/Qwen2.5-Coder-7B-Instruct"

        args_phi14 = parse_args(["--backbone", "phi4-14b"])
        assert args_phi14.base_model == "microsoft/Phi-4"

        args_phi_mini = parse_args(["--backbone", "phi4-mini"])
        assert args_phi_mini.base_model == "microsoft/Phi-4-mini-instruct"


class TestColabNotebookIntegrity:
    """Valida as fases 3, 4 e 5 no notebook Colab."""

    @pytest.fixture
    def notebook_data(self):
        nb_path = os.path.join(ROOT_DIR, "notebooks", "CyberRott_Colab_Training.ipynb")
        assert os.path.exists(nb_path), "Notebook não encontrado"
        with open(nb_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_cell_8_checkpoint_check(self, notebook_data):
        c8_text = "".join(notebook_data["cells"][8]["source"])
        assert ("LOCAL_GRPO_DIR = \"./CyberRott-v1.4-GRPO\"" in c8_text or "LOCAL_GRPO_DIR = \"./CyberRott-v1.3-GRPO\"" in c8_text)
        assert ("/content/drive/MyDrive/CyberRott_Checkpoints/CyberRott-v1.4-GRPO" in c8_text or "/content/drive/MyDrive/CyberRott_Checkpoints/CyberRott-v1.3-GRPO" in c8_text)
        assert ("/content/drive/MyDrive/CyberRott_Checkpoints/CyberRott_v1_4_GRPO" in c8_text or "/content/drive/MyDrive/CyberRott_Checkpoints/CyberRott_v1_3_GRPO" in c8_text)
        assert "RETREINO GRPO PULADO COM SUCESSO!" in c8_text
        assert "scripts/train_grpo.py" in c8_text

    def test_cell_10_priority_candidates(self, notebook_data):
        c10_text = "".join(notebook_data["cells"][10]["source"])
        assert ("FINAL_CHECKPOINT = \"./CyberRott-v1.4-GRPO\"" in c10_text or "FINAL_CHECKPOINT = \"./CyberRott-v1.3-GRPO\"" in c10_text)
        assert ("/content/drive/MyDrive/CyberRott_Checkpoints/CyberRott-v1.4-GRPO" in c10_text or "/content/drive/MyDrive/CyberRott_Checkpoints/CyberRott-v1.3-GRPO" in c10_text)

    def test_cell_12_model_name_update(self, notebook_data):
        c12_text = "".join(notebook_data["cells"][12]["source"])
        assert ("--model_name ./CyberRott-v1.4-GRPO" in c12_text or "--model_name ./CyberRott-v1.3-GRPO" in c12_text)
        assert "johnknevescarrijo/CyberRott-v1.2-GRPO" not in c12_text


class TestEvalMergeExportFixes:
    """Valida correções da Feature 4 em eval_wmdp_vllm, merge_lora, export_gguf e upload_to_hf."""

    def test_resolve_model_or_adapter_path_respects_explicit_local_path_first(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Cria candidato prioritário ./CyberRott-v1.4-GRPO e um modelo explícito ./meu-modelo-CyberRott-v1.3
        prio_dir = tmp_path / "CyberRott-v1.4-GRPO"
        prio_dir.mkdir()
        (prio_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

        explicit_dir = tmp_path / "meu-modelo-CyberRott-v1.3"
        explicit_dir.mkdir()
        (explicit_dir / "config.json").write_text("{}", encoding="utf-8")

        resolved_model, _, _ = resolve_model_or_adapter_path("./meu-modelo-CyberRott-v1.3")
        assert resolved_model == os.path.abspath(str(explicit_dir))

    def test_eval_main_handles_zero_questions_and_writes_versioned_latest(self, tmp_path, monkeypatch):
        import types
        from unittest.mock import MagicMock
        import scripts.eval_wmdp_vllm as eval_mod

        model_dir = tmp_path / "CyberRott-v1.4-Standalone"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")

        out_dir = tmp_path / "eval_out"

        class FakeLLM:
            def __init__(self, **kwargs):
                pass

            def generate(self, prompts, sampling_params):
                return []

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        fake_vllm.SamplingParams = MagicMock()

        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda *a, **kw: []

        fake_tok = MagicMock()
        fake_tok.apply_chat_template.return_value = "prompt"
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **kw: fake_tok)

        monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
        monkeypatch.setattr(eval_mod, "load_dataset", lambda *a, **kw: [])
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "eval_wmdp_vllm.py",
                "--model_name",
                str(model_dir),
                "--output_dir",
                str(out_dir),
            ],
        )

        eval_mod.main()

        summary = json.loads((out_dir / "latest_summaryv1.4.json").read_text(encoding="utf-8"))
        assert summary["total_questions"] == 0
        assert summary["accuracy"] == 0.0
        assert summary["stderr"] == 0.0
        assert (out_dir / "latest_full_resultsv1.4.json").exists()

    def test_merge_lora_no_silent_fallback_and_raises_file_not_found(self, tmp_path):
        from scripts.merge_lora import merge_and_save

        missing = str(tmp_path / "does_not_exist")
        assert resolve_adapter_path(missing) is None

        with pytest.raises(FileNotFoundError):
            merge_and_save("microsoft/Phi-4-mini-instruct", missing, str(tmp_path / "out"))

        valid_adapter = tmp_path / "valid_adapter"
        valid_adapter.mkdir()
        (valid_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            merge_and_save(
                "microsoft/Phi-4-mini-instruct",
                str(valid_adapter),
                str(tmp_path / "out"),
                sft_adapter_path=missing,
            )

    def test_export_gguf_build_modelfile_content_and_run_cmd_no_shell(self, monkeypatch):
        from types import SimpleNamespace
        from scripts.export_gguf import build_modelfile_content, run_cmd

        phi4_mf = build_modelfile_content("model.gguf", base_model="microsoft/Phi-4-mini-instruct")
        assert "<|system|>" in phi4_mf
        assert "<|user|>" in phi4_mf
        assert "<|assistant|>" in phi4_mf
        assert '<|end|>' in phi4_mf
        assert 'PARAMETER stop "<|end|>"' in phi4_mf

        qwen_mf = build_modelfile_content("model.gguf", base_model="Qwen/Qwen2.5-Coder-7B-Instruct")
        assert "<|im_start|>system" in qwen_mf
        assert "<|im_end|>" in qwen_mf

        observed = {}

        def fake_run(cmd, **kwargs):
            observed["cmd"] = cmd
            observed["shell"] = kwargs.get("shell")
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr("scripts.export_gguf.subprocess.run", fake_run)
        run_cmd(["echo", "ok"], "Test")
        assert observed["cmd"] == ["echo", "ok"]
        assert observed["shell"] is False

    def test_upload_to_hf_v14_fallback(self, tmp_path, monkeypatch):
        from scripts.upload_to_hf import DEFAULT_REPO_NAME, resolve_model_folder

        monkeypatch.chdir(tmp_path)
        assert DEFAULT_REPO_NAME == "CyberRott-v1.4-GRPO"
        assert resolve_model_folder(None) == "./CyberRott-v1.4-Standalone"
