"""CPU-only recovery/retention checks; no worker or OS process control."""

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from slime_ext.backends.megatron import rolling_checkpoint as rolling


class RollingCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.args = SimpleNamespace(
            rolling_checkpoint_retain_interval=100, save_interval=5,
            save=str(self.root), async_save=True, use_persistent_ckpt_worker=True, ckpt_format="torch_dist",
            no_save_optim=False, no_save_rng=False, rank=0,
            data_source_path="slime.data.data_source.RolloutDataSourceWithBuffer",
        )
        self.actor = SimpleNamespace(args=self.args, _rolling_async_initialized=True)
        self.actor.rollout_manager = SimpleNamespace(save=SimpleNamespace(remote=self.cursor))
        ray = ModuleType("ray")
        ray.get = lambda result: result
        ray_patch = patch.dict(sys.modules, {"ray": ray})
        ray_patch.start()
        self.addCleanup(ray_patch.stop)
        # Exercise exact cleanup deterministically in our temporary directory;
        # there are no background worker processes or real OS signals.
        class InlineThread:
            def __init__(self, target, args, **kwargs):
                self.target, self.args = target, args

            def start(self):
                self.target(*self.args)
        self.thread_patch = patch.object(rolling.threading, "Thread", InlineThread)
        self.thread_patch.start()
        self.addCleanup(self.thread_patch.stop)

    def context(self, iteration, final=False):
        captured = []
        wrapped = rolling._make_save_model(
            lambda *a, **kw: captured.append(rolling._active_save.get())
        )
        async_utils = ModuleType("megatron.training.async_utils")
        async_utils.maybe_finalize_async_save = lambda **kw: None
        with patch.dict(sys.modules, {"megatron.training.async_utils": async_utils}):
            wrapped(self.actor, iteration, force_sync=final)
        self.assertIsNone(rolling._active_save.get())
        return captured[0]

    def model(self, iteration, final=False):
        context = self.context(iteration, final)
        folder = self.root / f"iter_{iteration:07d}"
        folder.mkdir()
        for name in (".metadata", "common.pt", "__0_0.distcp"):
            (folder / name).write_bytes(b"complete native shard fixture")
        rolling._model_finished(context)

    def cursor(self, iteration, fail=False):
        def native_save(data_source, index):
            cursor = rolling._cursor_path(Path(data_source.args.save), index)
            cursor.parent.mkdir(exist_ok=True)
            cursor.write_bytes(b"partial" if fail else b"complete dataset state")
            if fail:
                raise OSError("simulated cursor disk failure")
        rolling._make_data_save(native_save)(self.actor, iteration)

    def latest(self):
        tracker = self.root / rolling._TRACKER
        return int(tracker.read_text()) if tracker.exists() else None

    def test_cursor_precedes_model_and_prior_survives_until_finalize(self):
        self.model(4)
        context = self.context(9)  # save_model captured the cursor before writing.
        self.assertTrue(rolling._cursor_path(self.root, 9).is_file())
        self.assertEqual(self.latest(), 4)
        self.assertTrue((self.root / "iter_0000004").exists())
        (self.root / "iter_0000009").mkdir()
        rolling._model_finished(context)
        self.assertEqual(self.latest(), 9)
        self.assertFalse((self.root / "iter_0000004").exists())
        self.assertFalse(rolling._cursor_path(self.root, 4).exists())

    def test_driver_duplicate_cursor_save_does_not_overwrite_snapshot(self):
        self.model(4)
        cursor = rolling._cursor_path(self.root, 4)
        before = cursor.read_bytes()
        self.cursor(4, fail=True)  # Would overwrite then raise if called twice.
        self.assertEqual(cursor.read_bytes(), before)
        self.assertEqual(self.latest(), 4)
        self.assertFalse(list(self.root.rglob("*.ready.json")))
        self.assertFalse((self.root / ".slime-checkpoint.lock").exists())

    def test_one_temporary_plus_every_hundred_and_final(self):
        for iteration in range(4, 205, 5):
            self.cursor(iteration)
            self.model(iteration)
        self.assertEqual(self.latest(), 204)
        self.assertEqual(sorted(p.name for p in self.root.glob("iter_*")),
                         ["iter_0000099", "iter_0000199", "iter_0000204"])
        self.model(207, final=True)
        self.assertEqual(self.latest(), 207)
        self.assertTrue(rolling._cursor_path(self.root, 207).is_file())
        self.assertTrue(json.loads((self.root / "iter_0000207" / rolling._COMPLETE).read_text())["retain"])
        self.assertFalse((self.root / "iter_0000204").exists())

    def test_failed_cursor_leaves_previous_recoverable(self):
        self.cursor(4)
        self.model(4)
        self.actor.rollout_manager.save.remote = lambda i: self.cursor(i, fail=True)
        with self.assertRaises(OSError):
            self.model(9)
        self.assertEqual(self.latest(), 4)
        self.assertFalse((self.root / "iter_0000009" / rolling._COMPLETE).exists())
        self.assertTrue((self.root / "iter_0000004").exists())

    def test_failed_native_writer_never_advances_or_prunes(self):
        self.cursor(4)
        self.model(4)
        self.cursor(9)
        context = self.context(9)
        def failing_native_finalize():
            raise OSError("simulated distributed writer failure")
        def iter_finalize_fn():
            self.fail("non-atomic native tracker callback should have been replaced")
        iter_finalize_fn.__module__ = "megatron.training.checkpointing"
        request = namedtuple("Request", "finalize_fns")([failing_native_finalize, iter_finalize_fn])
        token = rolling._active_save.set(context)
        try:
            captured = rolling._make_schedule(lambda r: r)(request)
        finally:
            rolling._active_save.reset(token)
        with self.assertRaises(OSError):
            for callback in captured.finalize_fns:
                callback()
        self.assertEqual(self.latest(), 4)
        self.assertTrue((self.root / "iter_0000004").exists())

    def test_historical_and_symlink_checkpoints_are_never_pruned(self):
        old = self.root / "iter_0000003"
        old.mkdir()
        (self.root / rolling._TRACKER).write_text("3")
        self.cursor(4)
        self.model(4)
        self.assertTrue(old.exists())
        linked = self.root / "iter_0000002"
        linked.symlink_to(self.root / "iter_0000004", target_is_directory=True)
        rolling._delete_previous(self.root, 2)
        self.assertTrue(linked.is_symlink())

    def test_failed_atomic_pointer_update_preserves_previous(self):
        self.cursor(4)
        self.model(4)
        self.cursor(9)
        original = rolling.os.replace
        def fail_tracker(source, destination):
            if Path(destination).name == rolling._TRACKER:
                raise OSError("simulated tracker commit failure")
            original(source, destination)
        with patch.object(rolling.os, "replace", fail_tracker):
            with self.assertRaises(OSError):
                self.model(9)
        self.assertEqual(self.latest(), 4)
        self.assertTrue((self.root / "iter_0000004").exists())

    def test_cpu_optimizer_snapshot_does_not_alias_training_storage(self):
        for view in (False, True):
            with self.subTest(noncontiguous_view=view):
                storage = torch.arange(16, dtype=torch.float32).share_memory_()
                state = storage.reshape(4, 4).t()[1:3] if view else storage
                expected = state.clone()
                writer = SimpleNamespace()
                planner = SimpleNamespace(resolve_data=lambda item, _state=state: _state)
                def native_prepare(writer, plan, planner):
                    saved = planner.resolve_data("optimizer")
                    # Native preparation clones only oversized CPU storage.
                    self.assertEqual(saved.untyped_storage().size(), saved.numel() * saved.itemsize)
                    writer.write_buckets = [("shard", "key", ([], [("optimizer", saved)]))]
                snapshot = rolling._make_prepare_write_data(native_prepare)
                token = rolling._active_save.set(self.context(4))
                try:
                    snapshot(writer, None, planner)
                finally:
                    rolling._active_save.reset(token)
                state.add_(100)
                saved = writer.write_buckets[0][2][1][0][1]
                self.assertTrue(torch.equal(saved, expected))
                self.assertNotEqual(saved.data_ptr(), state.data_ptr())
                self.assertTrue(saved.is_shared())
                self.assertTrue(saved.is_contiguous())

    def test_poll_is_nonblocking_and_disabled_mode_is_unchanged(self):
        calls = []
        async_utils = ModuleType("megatron.training.async_utils")
        async_utils.maybe_finalize_async_save = lambda **kw: calls.append(kw)
        async_utils.is_empty_async_queue = lambda: False
        wrapped = rolling._make_train(lambda *a, **kw: "trained")
        with patch.dict(sys.modules, {"megatron.training.async_utils": async_utils}):
            self.assertEqual(wrapped(self.actor, 5, None), "trained")
            self.args.rolling_checkpoint_retain_interval = 0
            self.assertEqual(wrapped(self.actor, 6, None), "trained")
        self.assertEqual(calls, [{"blocking": False}, {"blocking": False}])

    def test_completed_writer_closes_before_next_save_can_stage_buffers(self):
        calls = []
        state = {"pending": True}
        async_utils = ModuleType("megatron.training.async_utils")
        async_utils.is_empty_async_queue = lambda: not state["pending"]
        def finalize(**kwargs):
            if kwargs.get("terminate"):
                self.assertFalse(state["pending"], "must not wait for an unfinished writer")
            calls.append(kwargs)
        async_utils.maybe_finalize_async_save = finalize
        def train(*args, **kwargs):
            self.assertEqual(calls, [{"blocking": False}])
            state["pending"] = False  # Native completion becomes visible during training.
            return "trained"
        with patch.dict(sys.modules, {"megatron.training.async_utils": async_utils}):
            self.assertEqual(rolling._make_train(train)(self.actor, 5, None), "trained")
        self.assertEqual(calls, [{"blocking": False}, {"blocking": False},
                                 {"blocking": False, "terminate": True}])

    def test_persistent_writer_initializes_once_and_drains_at_final(self):
        calls = []
        async_utils = ModuleType("megatron.training.async_utils")
        async_utils.is_empty_async_queue = lambda: True
        async_utils.init_persistent_async_worker = lambda rank: calls.append(("initialize", rank))
        async_utils.maybe_finalize_async_save = lambda **kw: calls.append(kw)
        self.actor._rolling_async_initialized = False
        wrapped = rolling._make_save_model(lambda self, i, **kw: calls.append((i, kw["force_sync"])))
        with patch.dict(sys.modules, {"megatron.training.async_utils": async_utils}):
            wrapped(self.actor, 4)
            wrapped(self.actor, 9)
            wrapped(self.actor, 12, force_sync=True)
        drain = {"blocking": True, "terminate": True}
        self.assertEqual(calls, [("initialize", 0), drain, (4, False), drain, (9, False),
                                 drain, (12, True), drain])
        self.assertIsNone(rolling._active_save.get())

    def test_lagging_writer_closes_before_successor_snapshot(self):
        state = {"previous_writer_live": True}
        async_utils = ModuleType("megatron.training.async_utils")
        def finalize(**kwargs):
            self.assertEqual(kwargs, {"blocking": True, "terminate": True})
            state["previous_writer_live"] = False
        async_utils.maybe_finalize_async_save = finalize
        def save(actor, iteration, **kwargs):
            self.assertFalse(state["previous_writer_live"])
            return "new snapshot"
        with patch.dict(sys.modules, {"megatron.training.async_utils": async_utils}):
            self.assertEqual(rolling._make_save_model(save)(self.actor, 9), "new snapshot")

    def test_validation_rejects_incomplete_recovery_and_bad_cadence(self):
        rolling.validate_args(self.args)
        for key, value in [("save_interval", 6), ("no_save_optim", True),
                           ("no_save_rng", True), ("async_save", False), ("use_critic", True),
                           ("use_persistent_ckpt_worker", False)]:
            with self.subTest(key=key), patch.object(self.args, key, value, create=True):
                with self.assertRaises(ValueError):
                    rolling.validate_args(self.args)

    def test_launcher_default_and_save_opt_out(self):
        repo_root = Path(__file__).resolve().parents[2]
        launchers = [
            repo_root / "examples/qwen3_omni_rl/run.sh",
            repo_root / "examples/qwen3_omni_agent_gspo/run.sh",
        ]
        # Each launcher's DRY_RUN branch exits before data, Ray, tracking or
        # traps. This subprocess renders argv only and creates no children.
        for launcher in launchers:
            self.assertTrue(launcher.is_file(), f"missing launcher {launcher}")
            for extra, expected in [
                ({"LATEST_SAVE_INTERVAL": "5"}, ("5", "100")),
                ({"LATEST_SAVE_INTERVAL": "0"}, ("100", None)),
                ({"SAVE_INTERVAL": "0"}, (None, None)),
            ]:
                with self.subTest(launcher=launcher.parent.name, extra=extra):
                    env = {
                        **os.environ,
                        "DRY_RUN": "1",
                        "PYTHON_BIN": sys.executable,
                        "SAVE_INTERVAL": "100",
                        "LATEST_SAVE_INTERVAL": "5",
                        **extra,
                    }
                    output = subprocess.run(
                        ["bash", str(launcher)], env=env, check=True, text=True, capture_output=True
                    ).stdout
                    command = shlex.split(output)
                    for option, value in zip(("--save-interval", "--rolling-checkpoint-retain-interval"), expected, strict=True):
                        if value is None:
                            self.assertNotIn(option, command)
                        else:
                            self.assertEqual(command[command.index(option) + 1], value)
                    self.assertEqual("--async-save" in command, expected[1] is not None)
                    self.assertEqual("--use-persistent-ckpt-worker" in command, expected[1] is not None)
                    self.assertNotIn("--no-save-optim", command)
                    self.assertNotIn("--no-save-rng", command)


if __name__ == "__main__":
    unittest.main()
