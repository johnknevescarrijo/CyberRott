import os
import sys
import json
import tempfile
import pytest

# Adiciona o diretório raiz ao path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.eval_wmdp_vllm import find_checkpoint_subdir, extract_answer, resolve_model_or_adapter_path
from scripts.merge_lora import resolve_adapter_path


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
        assert "LOCAL_GRPO_DIR = \"./CyberRott-v1.3-GRPO\"" in c8_text
        assert "/content/drive/MyDrive/CyberRott_Checkpoints/CyberRott-v1.3-GRPO" in c8_text
        assert "/content/drive/MyDrive/CyberRott_Checkpoints/CyberRott_v1_3_GRPO" in c8_text
        assert "RETREINO GRPO PULADO COM SUCESSO!" in c8_text
        assert "scripts/train_grpo.py" in c8_text

    def test_cell_10_priority_candidates(self, notebook_data):
        c10_text = "".join(notebook_data["cells"][10]["source"])
        assert "FINAL_CHECKPOINT = \"./CyberRott-v1.3-GRPO\"" in c10_text
        assert "/content/drive/MyDrive/CyberRott_Checkpoints/CyberRott-v1.3-GRPO" in c10_text
        assert "/content/drive/MyDrive/CyberRott_Checkpoints/CyberRott_v1_3_GRPO" in c10_text

    def test_cell_12_model_name_update(self, notebook_data):
        c12_text = "".join(notebook_data["cells"][12]["source"])
        assert "--model_name ./CyberRott-v1.3-GRPO" in c12_text
        assert "johnknevescarrijo/CyberRott-v1.2-GRPO" not in c12_text
