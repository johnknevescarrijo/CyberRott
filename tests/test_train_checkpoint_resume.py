#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for automated SFT checkpoint inspection, resume, promotion,
and Google Drive synchronization in scripts/train_colab.py.
"""

import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import scripts.train_colab as train_colab


def _write_adapter_files(directory, weight_filename="adapter_model.safetensors", with_tokenizer=True):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 64, "lora_alpha": 128}),
        encoding="utf-8",
    )
    (directory / weight_filename).write_bytes(b"weights")
    if with_tokenizer:
        (directory / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "LlamaTokenizerFast"}),
            encoding="utf-8",
        )
        (directory / "tokenizer.json").write_text("{}", encoding="utf-8")


def _write_checkpoint(
    parent_dir,
    step,
    epoch,
    max_steps=879,
    num_train_epochs=3,
    with_adapter=True,
    with_heavy_state=True,
):
    ckpt_dir = parent_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if with_adapter:
        _write_adapter_files(ckpt_dir, with_tokenizer=True)
    if with_heavy_state:
        (ckpt_dir / "optimizer.pt").write_bytes(b"optimizer-heavy-bytes")
        (ckpt_dir / "scheduler.pt").write_bytes(b"scheduler-bytes")
        (ckpt_dir / "rng_state.pth").write_bytes(b"rng-bytes")
    (ckpt_dir / "trainer_state.json").write_text(
        json.dumps(
            {
                "epoch": epoch,
                "global_step": step,
                "max_steps": max_steps,
                "num_train_epochs": num_train_epochs,
            }
        ),
        encoding="utf-8",
    )
    return ckpt_dir


class TestCLICheckpointArguments:
    """Validate new checkpoint/Drive CLI flags and backwards compatibility."""

    def test_default_checkpoint_cli_args(self):
        args = train_colab.parse_args([])
        assert args.resume_from_checkpoint == "auto"
        assert args.drive_dir is None
        assert args.save_total_limit == 2
        assert args.dry_run is False

    def test_v1_4_preset_and_custom_checkpoint_flags(self, tmp_path):
        drive = tmp_path / "drive"
        args = train_colab.parse_args(
            [
                "--preset_v1_4",
                "--backbone",
                "qwen2.5-coder-7b",
                "--epochs",
                "4",
                "--data_path",
                "data/sample.jsonl",
                "--output_dir",
                str(tmp_path / "out"),
                "--resume_from_checkpoint",
                "none",
                "--drive_dir",
                str(drive),
                "--save_total_limit",
                "3",
                "--dry_run",
            ]
        )
        assert args.use_dora is True
        assert args.lora_r == 64
        assert args.lora_alpha == 128
        assert args.model_name_or_path == "Qwen/Qwen2.5-Coder-7B-Instruct"
        assert args.num_train_epochs == 4
        assert args.data_file == "data/sample.jsonl"
        assert args.resume_from_checkpoint == "none"
        assert args.drive_dir == str(drive)
        assert args.save_total_limit == 3
        assert args.dry_run is True


class TestCheckpointDetectionAndSorting:
    """Validate numeric checkpoint sorting and completion detection."""

    def test_selects_numerically_highest_checkpoint(self, tmp_path):
        out_dir = tmp_path / "CyberRott-v1.4-SFT-DoRA"
        _write_checkpoint(out_dir, step=90, epoch=0.3, max_steps=1200, num_train_epochs=3)
        ckpt_879 = _write_checkpoint(out_dir, step=879, epoch=2.2, max_steps=1200, num_train_epochs=3)

        listed = train_colab.list_checkpoints(str(out_dir))
        assert [s for s, _ in listed] == [90, 879]

        inspection = train_colab.inspect_checkpoint_state(
            output_dir=str(out_dir),
            resume_from_checkpoint="auto",
            num_train_epochs=3,
            drive_mount=str(tmp_path / "unmounted_drive"),
        )
        assert inspection["status"] == "resume"
        assert inspection["resume_checkpoint"] == str(ckpt_879)

    def test_completed_when_adapter_exists_at_root(self, tmp_path):
        out_dir = tmp_path / "CyberRott-v1.4-SFT-DoRA"
        _write_adapter_files(out_dir, weight_filename="adapter_model.bin")

        inspection = train_colab.inspect_checkpoint_state(
            output_dir=str(out_dir),
            resume_from_checkpoint="auto",
            num_train_epochs=3,
            drive_mount=str(tmp_path / "unmounted_drive"),
        )
        assert inspection["status"] == "completed"
        assert inspection["completed_source"] == str(out_dir)
        assert inspection["completed_in_checkpoint"] is False

    def test_completed_only_inside_checkpoint_promotes_without_heavy_files(self, tmp_path):
        out_dir = tmp_path / "CyberRott-v1.4-SFT-DoRA"
        ckpt_879 = _write_checkpoint(
            out_dir,
            step=879,
            epoch=3.0,
            max_steps=879,
            num_train_epochs=3,
            with_adapter=True,
            with_heavy_state=True,
        )

        inspection = train_colab.inspect_checkpoint_state(
            output_dir=str(out_dir),
            resume_from_checkpoint="auto",
            num_train_epochs=3,
            drive_mount=str(tmp_path / "unmounted_drive"),
        )
        assert inspection["status"] == "completed"
        assert inspection["completed_in_checkpoint"] is True
        assert inspection["completed_source"] == str(ckpt_879)

        prepared = train_colab.prepare_checkpoint_for_execution(inspection, str(out_dir))
        assert prepared["final_adapter_dir"] == str(out_dir)
        assert train_colab.has_complete_adapter(str(out_dir))
        assert (out_dir / "tokenizer_config.json").exists()
        assert not (out_dir / "optimizer.pt").exists()
        assert not (out_dir / "scheduler.pt").exists()
        assert not (out_dir / "rng_state.pth").exists()

    def test_epoch_tolerance_marks_checkpoint_completed(self, tmp_path):
        out_dir = tmp_path / "CyberRott-v1.4-SFT-DoRA"
        _write_checkpoint(out_dir, step=878, epoch=2.9995, max_steps=879, num_train_epochs=3)

        inspection = train_colab.inspect_checkpoint_state(
            output_dir=str(out_dir),
            resume_from_checkpoint="auto",
            num_train_epochs=3,
            drive_mount=str(tmp_path / "unmounted_drive"),
        )
        assert inspection["status"] == "completed"

    def test_partial_checkpoint_triggers_resume(self, tmp_path):
        out_dir = tmp_path / "CyberRott-v1.4-SFT-DoRA"
        ckpt_293 = _write_checkpoint(out_dir, step=293, epoch=1.0, max_steps=879, num_train_epochs=3)

        inspection = train_colab.inspect_checkpoint_state(
            output_dir=str(out_dir),
            resume_from_checkpoint="auto",
            num_train_epochs=3,
            drive_mount=str(tmp_path / "unmounted_drive"),
        )
        assert inspection["status"] == "resume"
        assert inspection["resume_checkpoint"] == str(ckpt_293)

    def test_resume_none_and_false_ignore_existing_checkpoints(self, tmp_path):
        out_dir = tmp_path / "CyberRott-v1.4-SFT-DoRA"
        _write_adapter_files(out_dir)
        _write_checkpoint(out_dir, step=879, epoch=3.0, max_steps=879, num_train_epochs=3)

        for flag in ("none", "false"):
            inspection = train_colab.inspect_checkpoint_state(
                output_dir=str(out_dir),
                resume_from_checkpoint=flag,
                num_train_epochs=3,
                drive_mount=str(tmp_path / "unmounted_drive"),
            )
            assert inspection["status"] == "scratch"
            assert inspection["resume_checkpoint"] is None


class TestGoogleDriveCheckpointSyncAndResume:
    """Validate Drive mirror lookup, Drive-to-local sync, and DriveSyncCallback."""

    def test_partial_checkpoint_in_drive_underscore_variant_syncs_locally(self, tmp_path):
        local_out = tmp_path / "CyberRott-v1.4-SFT-DoRA"
        local_out.mkdir()
        drive_root = tmp_path / "drive" / "MyDrive" / "CyberRott_Checkpoints"
        drive_run = drive_root / "CyberRott_v1_4_SFT_DoRA"
        _write_checkpoint(drive_run, step=293, epoch=1.0, max_steps=879, num_train_epochs=3)

        inspection = train_colab.inspect_checkpoint_state(
            output_dir=str(local_out),
            resume_from_checkpoint="auto",
            num_train_epochs=3,
            drive_dir=str(drive_root),
        )
        assert inspection["status"] == "resume"
        assert inspection["needs_sync_from_drive"] is True
        assert inspection["resume_checkpoint"] == str(local_out / "checkpoint-293")

        prepared = train_colab.prepare_checkpoint_for_execution(inspection, str(local_out))
        assert os.path.isdir(prepared["resume_checkpoint"])
        assert (local_out / "checkpoint-293" / "trainer_state.json").exists()

    def test_newer_drive_checkpoint_overrides_older_local_checkpoint(self, tmp_path):
        local_out = tmp_path / "CyberRott-v1.4-SFT-DoRA"
        _write_checkpoint(local_out, step=293, epoch=1.0, max_steps=879, num_train_epochs=3)

        drive_root = tmp_path / "drive" / "MyDrive" / "CyberRott_Checkpoints"
        drive_run = drive_root / "CyberRott-v1.4-SFT-DoRA"
        _write_checkpoint(drive_run, step=586, epoch=2.0, max_steps=879, num_train_epochs=3)

        inspection = train_colab.inspect_checkpoint_state(
            output_dir=str(local_out),
            resume_from_checkpoint="auto",
            num_train_epochs=3,
            drive_dir=str(drive_root),
        )
        assert inspection["status"] == "resume"
        assert inspection["needs_sync_from_drive"] is True
        assert inspection["resume_checkpoint"] == str(local_out / "checkpoint-586")

        train_colab.prepare_checkpoint_for_execution(inspection, str(local_out))
        assert (local_out / "checkpoint-586" / "adapter_config.json").exists()

    def test_completed_checkpoint_in_drive_promotes_locally_and_skips_retrain(self, tmp_path):
        local_out = tmp_path / "CyberRott-v1.4-SFT-DoRA"
        drive_root = tmp_path / "drive" / "MyDrive" / "CyberRott_Checkpoints"
        drive_run = drive_root / "CyberRott_v1_4_SFT_DoRA"
        _write_checkpoint(drive_run, step=879, epoch=3.0, max_steps=879, num_train_epochs=3)

        inspection = train_colab.inspect_checkpoint_state(
            output_dir=str(local_out),
            resume_from_checkpoint="auto",
            num_train_epochs=3,
            drive_dir=str(drive_root),
        )
        assert inspection["status"] == "completed"
        assert inspection["needs_sync_from_drive"] is True

        train_colab.prepare_checkpoint_for_execution(inspection, str(local_out))
        assert train_colab.has_complete_adapter(str(local_out))
        assert not (local_out / "optimizer.pt").exists()
        assert train_colab.has_complete_adapter(str(drive_root / "CyberRott-v1.4-SFT-DoRA"))

    def test_drive_sync_callback_syncs_and_enforces_save_total_limit(self, tmp_path):
        local_out = tmp_path / "CyberRott-v1.4-SFT-DoRA"
        drive_target = tmp_path / "drive" / "CyberRott_Checkpoints" / "CyberRott-v1.4-SFT-DoRA"

        cb = train_colab.DriveSyncCallback(drive_target_dir=str(drive_target), save_total_limit=2)
        args = SimpleNamespace(output_dir=str(local_out))

        for step in (100, 200, 300):
            _write_checkpoint(local_out, step=step, epoch=step / 100.0, max_steps=400, num_train_epochs=4)
            state = SimpleNamespace(global_step=step, is_world_process_zero=True)
            cb.on_save(args, state, control=None)

        drive_ckpts = [s for s, _ in train_colab.list_checkpoints(str(drive_target))]
        assert drive_ckpts == [200, 300]


class TestTrainExecutionWithMocks:
    """End-to-end orchestration tests for train() using lightweight mocks (no GPU)."""

    def _mock_hf_stack(self, monkeypatch, data_file):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<|endoftext|>"
        mock_tokenizer.eos_token = "<|endoftext|>"

        mock_base_model = MagicMock()
        mock_peft_model = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer_cls = MagicMock(return_value=mock_trainer)

        class DummyDataset:
            column_names = ["text"]

            def __len__(self):
                return 2

            def map(self, fn, batched=True):
                return self

        monkeypatch.setattr(train_colab.AutoTokenizer, "from_pretrained", MagicMock(return_value=mock_tokenizer))
        monkeypatch.setattr(train_colab.AutoModelForCausalLM, "from_pretrained", MagicMock(return_value=mock_base_model))
        monkeypatch.setattr(train_colab.PeftModel, "from_pretrained", MagicMock(return_value=mock_peft_model))
        monkeypatch.setattr(train_colab, "load_dataset", MagicMock(return_value=DummyDataset()))
        monkeypatch.setattr(train_colab, "SFTTrainer", mock_trainer_cls)
        monkeypatch.setattr(train_colab, "DEFAULT_DRIVE_MOUNT", str(data_file.parent / "unmounted_drive"))
        monkeypatch.setattr(train_colab, "DEFAULT_DRIVE_CHECKPOINTS_DIR", str(data_file.parent / "unmounted_drive_ckpts"))

        return {
            "tokenizer": mock_tokenizer,
            "base_model": mock_base_model,
            "peft_model": mock_peft_model,
            "trainer": mock_trainer,
            "trainer_cls": mock_trainer_cls,
        }

    def test_train_skips_retraining_when_completed_at_root(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "CyberRott-v1.4-SFT-DoRA"
        _write_adapter_files(out_dir)
        data_file = tmp_path / "train.jsonl"
        data_file.write_text('{"text": "sample"}\n', encoding="utf-8")

        mocks = self._mock_hf_stack(monkeypatch, data_file)
        args = train_colab.parse_args(
            [
                "--output_dir",
                str(out_dir),
                "--data_path",
                str(data_file),
                "--epochs",
                "3",
            ]
        )
        result = train_colab.train(args)

        assert result["status"] == "completed"
        assert result["skipped_training"] is True
        mocks["trainer"].train.assert_not_called()
        train_colab.PeftModel.from_pretrained.assert_called_once_with(mocks["base_model"], str(out_dir))

    def test_train_promotes_completed_checkpoint_and_skips_training(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "CyberRott-v1.4-SFT-DoRA"
        _write_checkpoint(out_dir, step=879, epoch=3.0, max_steps=879, num_train_epochs=3)
        data_file = tmp_path / "train.jsonl"
        data_file.write_text('{"text": "sample"}\n', encoding="utf-8")

        mocks = self._mock_hf_stack(monkeypatch, data_file)
        args = train_colab.parse_args(
            [
                "--output_dir",
                str(out_dir),
                "--data_path",
                str(data_file),
                "--epochs",
                "3",
            ]
        )
        result = train_colab.train(args)

        assert result["status"] == "completed"
        assert result["skipped_training"] is True
        assert train_colab.has_complete_adapter(str(out_dir))
        assert not (out_dir / "optimizer.pt").exists()
        mocks["trainer"].train.assert_not_called()
        train_colab.PeftModel.from_pretrained.assert_called_once_with(mocks["base_model"], str(out_dir))

    def test_train_resumes_from_partial_checkpoint(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "CyberRott-v1.4-SFT-DoRA"
        ckpt_293 = _write_checkpoint(out_dir, step=293, epoch=1.0, max_steps=879, num_train_epochs=3)
        data_file = tmp_path / "train.jsonl"
        data_file.write_text('{"text": "sample"}\n', encoding="utf-8")

        mocks = self._mock_hf_stack(monkeypatch, data_file)
        drive_root = tmp_path / "drive" / "CyberRott_Checkpoints"
        args = train_colab.parse_args(
            [
                "--output_dir",
                str(out_dir),
                "--data_path",
                str(data_file),
                "--epochs",
                "3",
                "--drive_dir",
                str(drive_root),
                "--save_total_limit",
                "3",
            ]
        )
        result = train_colab.train(args)

        assert result["status"] == "trained"
        assert result["resumed_from"] == str(ckpt_293)
        mocks["trainer"].train.assert_called_once_with(resume_from_checkpoint=str(ckpt_293))
        mocks["trainer"].save_model.assert_called_once_with(str(out_dir))
        mocks["trainer"].add_callback.assert_called_once()
        sft_cfg = mocks["trainer_cls"].call_args.kwargs["args"]
        assert sft_cfg.save_total_limit == 3

    def test_train_with_resume_none_starts_from_scratch(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "CyberRott-v1.4-SFT-DoRA"
        _write_checkpoint(out_dir, step=293, epoch=1.0, max_steps=879, num_train_epochs=3)
        data_file = tmp_path / "train.jsonl"
        data_file.write_text('{"text": "sample"}\n', encoding="utf-8")

        mocks = self._mock_hf_stack(monkeypatch, data_file)
        args = train_colab.parse_args(
            [
                "--output_dir",
                str(out_dir),
                "--data_path",
                str(data_file),
                "--epochs",
                "3",
                "--resume_from_checkpoint",
                "none",
            ]
        )
        result = train_colab.train(args)

        assert result["status"] == "trained"
        assert result["resumed_from"] is None
        mocks["trainer"].train.assert_called_once_with(resume_from_checkpoint=None)

    def test_train_dry_run_skips_model_downloads(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "CyberRott-v1.4-SFT-DoRA"
        _write_checkpoint(out_dir, step=879, epoch=3.0, max_steps=879, num_train_epochs=3)
        data_file = tmp_path / "train.jsonl"
        data_file.write_text('{"text": "sample"}\n', encoding="utf-8")

        mocks = self._mock_hf_stack(monkeypatch, data_file)
        args = train_colab.parse_args(
            [
                "--output_dir",
                str(out_dir),
                "--data_path",
                str(data_file),
                "--dry_run",
            ]
        )
        result = train_colab.train(args)

        assert result["status"] == "completed"
        assert result["dry_run"] is True
        train_colab.AutoModelForCausalLM.from_pretrained.assert_not_called()
        train_colab.PeftModel.from_pretrained.assert_not_called()
        mocks["trainer"].train.assert_not_called()

    def test_patch_checkpoint_batch_size_updates_before_sft_resume(self, tmp_path, monkeypatch, capsys):
        out_dir = tmp_path / "CyberRott-v1.4-SFT-DoRA"
        ckpt_100 = _write_checkpoint(out_dir, step=100, epoch=0.5, max_steps=879, num_train_epochs=3)
        state_path = ckpt_100 / "trainer_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["train_batch_size"] = 2
        state_path.write_text(json.dumps(state), encoding="utf-8")

        drive_root = tmp_path / "drive" / "CyberRott_Checkpoints"
        drive_run = drive_root / "CyberRott_v1_4_SFT_DoRA"
        drive_ckpt_100 = _write_checkpoint(drive_run, step=100, epoch=0.5, max_steps=879, num_train_epochs=3)
        drive_state_path = drive_ckpt_100 / "trainer_state.json"
        drive_state = json.loads(drive_state_path.read_text(encoding="utf-8"))
        drive_state["train_batch_size"] = 2
        drive_state_path.write_text(json.dumps(drive_state), encoding="utf-8")

        # Direct unit check on Drive + local checkpoint
        assert train_colab.patch_checkpoint_batch_size(
            str(ckpt_100),
            1,
            output_dir=str(out_dir),
            drive_dir=str(drive_root),
        ) is True
        assert json.loads(state_path.read_text(encoding="utf-8"))["train_batch_size"] == 1
        assert json.loads(drive_state_path.read_text(encoding="utf-8"))["train_batch_size"] == 1
        assert "[VRAM Safeguard] Atualizado train_batch_size (2 -> 1)" in capsys.readouterr().out

        # Reset local checkpoint-100 to 2 and verify train() patches it before trainer.train(...)
        state["train_batch_size"] = 2
        state_path.write_text(json.dumps(state), encoding="utf-8")

        data_file = tmp_path / "train.jsonl"
        data_file.write_text('{"text": "sample"}\n', encoding="utf-8")
        mocks = self._mock_hf_stack(monkeypatch, data_file)
        observed = {}

        def _check_before_train(resume_from_checkpoint=None):
            observed["resume_from_checkpoint"] = resume_from_checkpoint
            observed["disk_train_batch_size"] = json.loads(state_path.read_text(encoding="utf-8"))["train_batch_size"]

        mocks["trainer"].train.side_effect = _check_before_train

        args = train_colab.parse_args(
            [
                "--output_dir",
                str(out_dir),
                "--data_path",
                str(data_file),
                "--epochs",
                "3",
                "--per_device_train_batch_size",
                "1",
                "--drive_dir",
                str(drive_root),
            ]
        )
        train_colab.train(args)

        assert observed["resume_from_checkpoint"] == str(ckpt_100)
        assert observed["disk_train_batch_size"] == 1
