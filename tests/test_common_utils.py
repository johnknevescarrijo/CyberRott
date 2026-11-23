#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for scripts/common.py shared utilities, script re-exports,
and safe GPU/vLLM process cleanup.
"""

import os
import signal
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import scripts.common as common
import scripts.merge_lora as merge_lora
import scripts.train_colab as train_colab
import scripts.train_grpo as train_grpo
import scripts.train_star as train_star


class TestCommonReexports:
    """Ensure all training/merge scripts reuse scripts/common.py and preserve backwards compatibility."""

    def test_shared_backbone_and_lora_helpers_are_reused(self):
        for mod in (train_colab, train_grpo, train_star, merge_lora):
            assert mod.SUPPORTED_BACKBONES is common.SUPPORTED_BACKBONES
            assert mod.resolve_backbone_config is common.resolve_backbone_config

        for mod in (train_colab, train_grpo, train_star):
            assert mod.build_lora_config is common.build_lora_config
            assert mod.setup_hardware_optimizations is common.setup_hardware_optimizations

    def test_shared_checkpoint_and_drive_helpers_are_reused(self):
        for attr in (
            "DEFAULT_DRIVE_MOUNT",
            "DEFAULT_DRIVE_CHECKPOINTS_DIR",
            "ADAPTER_WEIGHT_FILES",
            "HEAVY_TRAINING_STATE_PREFIXES",
            "CHECKPOINT_ARTIFACT_HINT_FILES",
            "_SCRATCH_RESUME_VALUES",
            "has_complete_adapter",
            "list_checkpoints",
            "read_trainer_state",
            "is_checkpoint_completed",
            "_is_resumable_checkpoint_dir",
            "_get_drive_subfolder_variants",
            "resolve_drive_paths",
            "promote_adapter_files",
            "_patch_trainer_state_batch_size",
            "patch_checkpoint_batch_size",
        ):
            assert getattr(train_colab, attr) is getattr(common, attr)
            assert getattr(train_grpo, attr) is getattr(common, attr)

        assert train_colab.inspect_checkpoint_state is common.inspect_checkpoint_state
        assert train_colab.prepare_checkpoint_for_execution is common.prepare_checkpoint_for_execution
        assert train_colab.sync_checkpoint_to_google_drive is common.sync_checkpoint_to_google_drive
        assert train_colab.DriveSyncCallback is common.DriveSyncCallback

        assert callable(train_grpo.inspect_grpo_checkpoint_state)
        assert callable(train_grpo.prepare_grpo_checkpoint_for_execution)
        assert issubclass(train_grpo.GRPODriveSyncCallback, common.DriveSyncCallback)


class TestSafeVllmCleanup:
    """Validate that cleanup_external_vllm_server only SIGKILLs vLLM processes."""

    def test_is_vllm_process_inspects_proc_cmdline_and_comm(self, tmp_path):
        proc_dir = tmp_path / "proc"
        (proc_dir / "1001").mkdir(parents=True)
        (proc_dir / "1001" / "cmdline").write_bytes(b"python\x00-m\x00trl.scripts.vllm_serve\x00--port\x008000")
        (proc_dir / "1001" / "comm").write_text("python3", encoding="utf-8")

        (proc_dir / "1002").mkdir(parents=True)
        (proc_dir / "1002" / "cmdline").write_bytes(b"/usr/bin/python3\x00")
        (proc_dir / "1002" / "comm").write_text("VLLM::EngineCore", encoding="utf-8")

        (proc_dir / "1003").mkdir(parents=True)
        (proc_dir / "1003" / "cmdline").write_bytes(b"/usr/local/bin/ollama\x00serve")
        (proc_dir / "1003" / "comm").write_text("ollama", encoding="utf-8")

        (proc_dir / "1004").mkdir(parents=True)
        (proc_dir / "1004" / "cmdline").write_bytes(b"python\x00other_job.py")
        (proc_dir / "1004" / "comm").write_text("python", encoding="utf-8")

        assert common._is_vllm_process(1001, proc_root=str(proc_dir)) is True
        assert common._is_vllm_process(1002, proc_root=str(proc_dir)) is True
        assert common._is_vllm_process(1003, proc_root=str(proc_dir)) is False
        assert common._is_vllm_process(1004, proc_root=str(proc_dir)) is False
        assert common._is_vllm_process(9999, proc_root=str(proc_dir)) is False

    def test_cleanup_preserves_non_vllm_gpu_processes_and_ancestors(self, tmp_path):
        proc_dir = tmp_path / "proc"
        entries = {
            23116: (b"python\x00-m\x00trl.scripts.vllm_serve", "python3"),
            24501: (b"vllm.entrypoints.openai.api_server", "VLLM::EngineCore"),
            25001: (b"/usr/local/bin/ollama\x00serve", "ollama"),
            25002: (b"python\x00other_job.py", "python"),
            999991: (b"python\x00-m\x00vllm.entrypoints", "vllm"),
        }
        for pid, (cmdline, comm) in entries.items():
            p = proc_dir / str(pid)
            p.mkdir(parents=True)
            (p / "cmdline").write_bytes(cmdline)
            (p / "comm").write_text(comm, encoding="utf-8")

        self_pid = os.getpid()
        parent_pid = os.getppid()
        protected_set = {self_pid, parent_pid, 999991}

        def fake_run(cmd, **kwargs):
            if cmd[0] == "nvidia-smi":
                return SimpleNamespace(
                    returncode=0,
                    stdout=f"23116\n{self_pid}\n{parent_pid}\n999991\n25001\n25002\n24501\n",
                )
            return SimpleNamespace(returncode=1, stdout="")

        mock_kill = MagicMock()
        mock_cuda = SimpleNamespace(
            is_available=lambda: False,
            empty_cache=MagicMock(),
            mem_get_info=lambda: (0, 0),
        )

        killed = common.cleanup_external_vllm_server(
            protected_pids_fn=lambda: protected_set,
            subprocess_run=fake_run,
            os_kill=mock_kill,
            time_sleep=lambda *_: None,
            cuda_mod=mock_cuda,
            proc_root=str(proc_dir),
        )
        assert killed is True

        killed_args = [call.args for call in mock_kill.call_args_list]
        assert (23116, signal.SIGKILL) in killed_args
        assert (24501, signal.SIGKILL) in killed_args

        killed_pids = {pid for pid, _ in killed_args}
        assert 25001 not in killed_pids  # ollama preserved
        assert 25002 not in killed_pids  # other_job.py preserved
        assert self_pid not in killed_pids
        assert parent_pid not in killed_pids
        assert 999991 not in killed_pids

    def test_cleanup_when_only_non_vllm_gpu_apps_running_returns_false(self, tmp_path):
        proc_dir = tmp_path / "proc"
        for pid, cmd in ((30001, b"ollama\x00serve"), (30002, b"python\x00other_job.py")):
            p = proc_dir / str(pid)
            p.mkdir(parents=True)
            (p / "cmdline").write_bytes(cmd)
            (p / "comm").write_text("app", encoding="utf-8")

        def fake_run(cmd, **kwargs):
            if cmd[0] == "nvidia-smi":
                return SimpleNamespace(returncode=0, stdout="30001\n30002\n")
            return SimpleNamespace(returncode=1, stdout="")

        mock_kill = MagicMock()
        killed = common.cleanup_external_vllm_server(
            protected_pids_fn=lambda: {os.getpid()},
            subprocess_run=fake_run,
            os_kill=mock_kill,
            time_sleep=lambda *_: None,
            proc_root=str(proc_dir),
        )
        assert killed is False
        mock_kill.assert_not_called()
