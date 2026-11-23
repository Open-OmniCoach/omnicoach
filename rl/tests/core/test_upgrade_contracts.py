"""Static AST import audit and upstream slime-omni.patch EPD/BatchBuilder contract tests."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SLIME_ROOT = Path(os.environ.get("SLIME_DIR", str(REPO_ROOT / "third_party" / "slime")))
EXT_ROOT = REPO_ROOT / "slime_ext"

for _p in (str(SLIME_ROOT), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_module_from_path(name: str, rel_path: str | Path):
    target = Path(rel_path) if isinstance(rel_path, Path) else (REPO_ROOT / rel_path)
    spec = importlib.util.spec_from_file_location(name, target)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StaticUpgradeAuditTests(unittest.TestCase):
    def test_all_slime_imports_in_slime_ext_resolve_against_pinned_upstream(self):
        def module_file(modname: str) -> Path | None:
            p = SLIME_ROOT.joinpath(*modname.split("."))
            if p.with_suffix(".py").is_file():
                return p.with_suffix(".py")
            if (p / "__init__.py").is_file():
                return p / "__init__.py"
            return None

        errors = []
        for py in sorted(EXT_ROOT.rglob("*.py")):
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "slime" or alias.name.startswith("slime."):
                            if module_file(alias.name) is None:
                                errors.append(f"{py}:{node.lineno}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module and (node.module == "slime" or node.module.startswith("slime.")):
                        target = module_file(node.module)
                        if target is None:
                            errors.append(
                                f"{py}:{node.lineno}: from {node.module} import {[a.name for a in node.names]}"
                            )
                        else:
                            mod_src = target.read_text(encoding="utf-8")
                            for alias in node.names:
                                if alias.name == "*":
                                    continue
                                if module_file(f"{node.module}.{alias.name}") is None and alias.name not in mod_src:
                                    errors.append(
                                        f"{py}:{node.lineno}: missing symbol {alias.name} in {node.module}"
                                    )
        self.assertEqual(errors, [], "Broken slime.* imports:\n" + "\n".join(errors))

    def test_obsolete_modules_deleted_and_unreferenced(self):
        deleted_files = [
            "slime_ext/train_async.py",
            "slime_ext/slime",
            "slime_ext/slime_plugins",
            "slime_ext/sglang_processors",
            "slime_ext/examples",
            "slime_ext/tools",
            "scripts",
            "tools/bench",
        ]
        for rel in deleted_files:
            self.assertFalse((REPO_ROOT / rel).exists(), f"Obsolete file still exists: {rel}")

        banned_identifiers = {
            "patch_packed_mrope",
            "cp_reorder",
            "packed_rope",
            "sglang_tokenizer_workers",
            "train_async",
            "rollout_global_dataset",
        }
        hits = []
        for py in sorted(EXT_ROOT.rglob("*.py")):
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id in banned_identifiers:
                    hits.append(f"{py}:{node.lineno}: Name {node.id}")
                elif isinstance(node, ast.Attribute) and node.attr in banned_identifiers:
                    hits.append(f"{py}:{node.lineno}: Attr {node.attr}")
                elif isinstance(node, ast.FunctionDef) and node.name in banned_identifiers:
                    hits.append(f"{py}:{node.lineno}: Def {node.name}")
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    for part in mod.split("."):
                        if part in banned_identifiers:
                            hits.append(f"{py}:{node.lineno}: ImportFrom {mod}")
                    for alias in node.names:
                        if alias.name in banned_identifiers:
                            hits.append(f"{py}:{node.lineno}: ImportFrom {alias.name}")
        self.assertEqual(hits, [], "Stale references found:\n" + "\n".join(hits))


class UpstreamPatchHooksTests(unittest.TestCase):
    def _make_fake_ray(self):
        fake_ray = ModuleType("ray")
        fake_ray.get = lambda x, **kw: x
        fake_ray.put = lambda x, **kw: x

        def _remote(*dargs, **dkwargs):
            if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
                cls = dargs[0]
                cls.__ray_metadata__ = SimpleNamespace(modified_class=cls)
                return cls

            def decorator(cls):
                cls.__ray_metadata__ = SimpleNamespace(modified_class=cls)
                return cls

            return decorator

        fake_ray.remote = _remote
        fake_ray.method = lambda *a, **kw: (lambda fn: fn)
        fake_ray.util = ModuleType("ray.util")
        fake_ray.util.scheduling_strategies = ModuleType("ray.util.scheduling_strategies")
        fake_ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy = object
        fake_ray.actor = ModuleType("ray.actor")
        fake_ray.actor.ActorHandle = object
        fake_wandb = ModuleType("wandb")
        fake_tensorboard = ModuleType("torch.utils.tensorboard")
        fake_tensorboard.SummaryWriter = object
        return {
            "ray": fake_ray,
            "ray.util": fake_ray.util,
            "ray.util.scheduling_strategies": fake_ray.util.scheduling_strategies,
            "ray.actor": fake_ray.actor,
            "wandb": fake_wandb,
            "torch.utils.tensorboard": fake_tensorboard,
        }

    def test_epd_encoder_exclusion_in_both_rollout_and_serving(self):
        def make_group(worker_type, engines, *, gpu_count, gpu_offset, num_new):
            return SimpleNamespace(
                worker_type=worker_type,
                engines=engines,
                num_gpus_per_engine=gpu_count,
                gpu_offset=gpu_offset,
                num_new_engines=num_new,
                parallel_config=lambda: {"worker_type": worker_type},
            )

        encoder = make_group("encoder", ["enc-0", "enc-1"], gpu_count=1, gpu_offset=0, num_new=2)
        regular = make_group("regular", ["llm-0", "llm-1"], gpu_count=2, gpu_offset=2, num_new=1)
        server = SimpleNamespace(update_weights=True, server_groups=[encoder, regular])

        with patch.dict(sys.modules, self._make_fake_ray()):
            rollout_mod = importlib.import_module("slime.ray.rollout")
            serving_mod = importlib.import_module("slime.ray.serving")

            rm_cls = rollout_mod.RolloutManager.__ray_metadata__.modified_class
            rm = rm_cls.__new__(rm_cls)
            rm.serving = None
            rm.servers = {"actor": server}
            rm.rollout_engine_lock = object()
            engines, lock, num_new, gpu_counts, gpu_offsets, parallel_configs = (
                rm.get_updatable_engines_and_lock()
            )
            self.assertEqual(engines, ["llm-0", "llm-1"])
            self.assertIs(lock, rm.rollout_engine_lock)
            self.assertEqual(num_new, 1)
            self.assertEqual(gpu_counts, [2, 2])
            self.assertEqual(gpu_offsets, [2, 4])
            self.assertEqual(parallel_configs, [{"worker_type": "regular"}, {"worker_type": "regular"}])

            sc_cls = serving_mod.ServingCluster.__ray_metadata__.modified_class
            sc = sc_cls.__new__(sc_cls)
            sc.servers = {"actor": server}
            sc.lock = object()
            s_engines, s_lock, s_num_new, s_gpu_counts, s_gpu_offsets, s_cfgs = (
                sc.get_updatable_engines_and_lock()
            )
            self.assertEqual(s_engines, ["llm-0", "llm-1"])
            self.assertIs(s_lock, sc.lock)
            self.assertEqual(s_num_new, 1)
            self.assertEqual(s_gpu_counts, [2, 2])
            self.assertEqual(s_gpu_offsets, [2, 4])
            self.assertEqual(s_cfgs, [{"worker_type": "regular"}, {"worker_type": "regular"}])

    def test_engine_group_heterogeneous_host_ports_and_encoder_bootstrap_port(self):
        with patch.dict(sys.modules, self._make_fake_ray()):
            eg_mod = _load_module_from_path(
                "test_slime_engine_group_epd",
                SLIME_ROOT / "slime/backends/sglang_utils/engine_group.py",
            )

            class RemoteMethod:
                def __init__(self, host):
                    self.host = host

                def remote(self, *, start_port=None, consecutive=None):
                    del consecutive
                    return self.host, start_port or 20000

            class Engine:
                def __init__(self, host):
                    self._get_current_node_ip_and_free_port = RemoteMethod(host)

            args = SimpleNamespace(num_gpus_per_node=8, rollout_num_gpus_per_engine=2, sglang_dp_size=1)
            engines = [(2, Engine("node-a")), (3, Engine("node-a")), (4, Engine("node-a")), (5, Engine("node-b"))]
            allocated, cursors = eg_mod._allocate_rollout_engine_addr_and_ports_normal(
                args=args,
                rollout_engines=engines,
                worker_type="regular",
                allocate_encoder_bootstrap=True,
                num_gpus_per_engine=2,
                rank_offset=2,
            )
            self.assertEqual([allocated[r]["host"] for r in range(2, 6)], ["node-a", "node-a", "node-a", "node-b"])
            self.assertEqual([allocated[r]["encoder_bootstrap_port"] for r in (2, 3, 4)], [15002, 15036, 15070])
            self.assertEqual(allocated[5]["encoder_bootstrap_port"], 15002)
            self.assertEqual(cursors, {"node-a": 15102, "node-b": 15034})

    def test_batch_builder_propagates_turn_gspo_fields(self):
        fake_ray_mods = self._make_fake_ray()
        with patch.dict(sys.modules, fake_ray_mods):
            bb_mod = _load_module_from_path(
                "test_slime_batch_builder_turn_gspo",
                SLIME_ROOT / "slime/data/batch_builder.py",
            )
            args = SimpleNamespace(
                custom_reward_post_process_path=None,
                custom_convert_samples_to_train_data_path=None,
                rollout_data_transport="object-store",
                global_batch_size=2,
                micro_batch_size=1,
                use_dynamic_batch_size=False,
                balance_data=False,
                balance_by_flops=False,
                n_samples_per_prompt=2,
                rollout_batch_size=1,
                data_pad_size_multiplier=1,
                max_tokens_per_gpu=None,
                log_probs_max_tokens_per_gpu=None,
                use_dynamic_global_batch_size=False,
                use_rollout_routing_replay=False,
                use_score_centering=False,
            )
            builder = bb_mod.BatchBuilder(args)
            builder.rollout_id = 3
            builder.train_parallel_config = {
                "dp_size": 2,
                "cp_size": 1,
                "tp_size": 1,
                "vpp_size": 1,
                "microbatch_group_size_per_vp_stage": 1,
            }

            data = {
                "tokens": [[1, 2, 3, 4], [5, 6, 7, 8]],
                "response_lengths": [2, 2],
                "rewards": [1.0, -1.0],
                "raw_reward": [1.0, 0.0],
                "truncated": [0, 0],
                "loss_masks": [[1, 1], [1, 1]],
                "sample_indices": [0, 1],
                "rollout_ids": [0, 1],
                "rollout_mask_sums": [2, 2],
                "token_advantages": [[0.5, 1.5], [-0.5, -1.5]],
                "turn_ids": [[0, 1], [0, -1]],
                "turn_weights": [[0.25, 0.75], [1.0, 0.0]],
            }
            rank_boxes = builder.split_by_dp(data)
            self.assertEqual(len(rank_boxes), 2)
            rank0 = rank_boxes[0].inner
            self.assertIn("token_advantages", rank0)
            self.assertIn("turn_ids", rank0)
            self.assertIn("turn_weights", rank0)
            self.assertEqual(rank0["token_advantages"][0].dtype, torch.float32)
            self.assertEqual(rank0["turn_ids"][0].dtype, torch.long)
            self.assertEqual(rank0["turn_weights"][0].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
