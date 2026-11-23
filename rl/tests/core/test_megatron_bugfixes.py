"""CPU regression tests for confirmed Qwen3-Omni, Megatron, SGLang, and runtime fixes."""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SLIME_ROOT = Path(os.environ.get("SLIME_DIR", str(REPO_ROOT / "third_party" / "slime")))

for _p in (str(SLIME_ROOT), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_module_from_path(name: str, rel_path: str | Path):
    target = Path(rel_path) if isinstance(rel_path, Path) else (REPO_ROOT / rel_path)
    spec = importlib.util.spec_from_file_location(name, target)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CoreBugfixRegressionTests(unittest.TestCase):
    def test_bug1_empty_row_indices_loss_backward_and_lazy_tensor_ref_topp(self):
        from slime_ext.backends.megatron import loss as ext_loss

        fake_mpu = SimpleNamespace(
            get_context_parallel_world_size=lambda: 1,
            get_tensor_model_parallel_rank=lambda: 0,
            get_tensor_model_parallel_group=lambda: None,
        )
        args = SimpleNamespace(
            response_only_log_probs=True,
            response_only_lm_head=False,
            allgather_cp=False,
            rollout_temperature=1.0,
            entropy_coef=0.0,
            log_probs_chunk_size=16,
        )
        logits = torch.randn(1, 5, 11, dtype=torch.float32, requires_grad=True)
        tokens = [torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)]

        with patch.object(ext_loss, "mpu", fake_mpu):
            _, res = ext_loss.get_log_probs_and_entropy(
                logits,
                args=args,
                unconcat_tokens=tokens,
                total_lengths=[5],
                response_lengths=[0],
                with_entropy=True,
            )
            self.assertEqual(len(res["log_probs"]), 1)
            self.assertEqual(res["log_probs"][0].numel(), 0)
            (res["log_probs"][0].sum() + 0.0 * logits.sum()).backward()
            self.assertIsNotNone(logits.grad)
            self.assertTrue(torch.all(logits.grad == 0))

            class LazyRef:
                def __init__(self, data):
                    self._data = data
                    self.slices = []

                def load(self):
                    return self._data

                def __getitem__(self, item):
                    if not isinstance(item, slice):
                        raise TypeError("TensorRef only supports slice reads")
                    self.slices.append((item.start, item.stop))
                    return self._data[item]

                def __iter__(self):
                    raise AssertionError("list(TensorRef) must never be called")

            ids_ref = LazyRef(torch.tensor([1, 4, 2, 7, 9], dtype=torch.int32))
            offsets_ref = LazyRef(torch.tensor([0, 2, 5], dtype=torch.int32))
            keep = ext_loss._build_response_topp_keep_mask(
                response_lengths=[2],
                vocab_local=10,
                device=torch.device("cpu"),
                top_p_token_ids=[ids_ref],
                top_p_token_offsets=[offsets_ref],
            )
            self.assertEqual(ids_ref.slices, [(0, 5)])
            self.assertEqual(keep[0].nonzero(as_tuple=False).flatten().tolist(), [1, 4])
            self.assertEqual(keep[1].nonzero(as_tuple=False).flatten().tolist(), [2, 7, 9])

    def test_bug2_and_mcore20_text_model_cp1_zero_response_and_return_raw_freqs(self):
        class StubMRE(torch.nn.Module):
            pass

        class StubLayer(torch.nn.Module):
            def forward(self, hidden_states, **kwargs):
                return hidden_states, None

        class StubGPT(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.received_rows = []

            def _postprocess(self, hidden_states, **kwargs):
                self.received_rows.append(hidden_states.size(0))
                assert hidden_states.size(0) > 0, "Zero-row GEMM must never reach TE output_layer!"
                return hidden_states.transpose(0, 1) * 2.0

        fake_mods = {}
        for name in (
            "megatron",
            "megatron.core",
            "megatron.core.models",
            "megatron.core.models.common",
            "megatron.core.models.common.embeddings",
            "megatron.core.models.common.embeddings.rope_utils",
            "megatron.core.models.common.embeddings.rotary_pos_embedding",
            "megatron.core.models.gpt",
            "megatron.core.models.gpt.gpt_model",
            "megatron.core.tensor_parallel",
            "megatron.core.tensor_parallel.mappings",
            "megatron.core.transformer",
            "megatron.core.transformer.spec_utils",
            "megatron.core.transformer.transformer_block",
            "megatron.core.transformer.transformer_layer",
            "megatron.core.utils",
        ):
            fake_mods[name] = ModuleType(name)
        fake_mods["megatron.core.models.common.embeddings.rope_utils"].get_pos_emb_on_this_cp_rank = lambda emb, d, g: emb
        fake_mods["megatron.core.models.common.embeddings.rotary_pos_embedding"].MultimodalRotaryEmbedding = StubMRE
        fake_mods["megatron.core.models.gpt.gpt_model"].GPTModel = StubGPT
        fake_mods["megatron.core.models.gpt"].GPTModel = StubGPT
        fake_mods["megatron.core.tensor_parallel.mappings"].gather_from_sequence_parallel_region = lambda x, **kw: x
        fake_mods["megatron.core.tensor_parallel"].gather_from_sequence_parallel_region = lambda x, **kw: x
        fake_mods["megatron.core.transformer.transformer_block"].TransformerBlockSubmodules = type("TBS", (), {})
        fake_mods["megatron.core.transformer.transformer_layer"].TransformerLayer = StubLayer
        fake_mods["megatron.core.utils"].make_viewless_tensor = lambda inp, **kw: inp

        with patch.dict(sys.modules, fake_mods):
            tm = _load_module_from_path(
                "test_text_model_cpu",
                "slime_ext/models/qwen3_omni/text.py",
            )
            rope = tm.Qwen3OmniMultimodalRotaryEmbedding.__new__(tm.Qwen3OmniMultimodalRotaryEmbedding)
            torch.nn.Module.__init__(rope)
            rope.inv_freq = torch.ones(4)
            rope.rotary_interleaved = False
            rope.seq_len_interpolation_factor = None
            rope.cp_group = None
            pos = torch.tensor([[[10]], [[20]], [[30]]])
            out = rope(pos, [2, 1, 1], return_raw_freqs=False)
            self.assertEqual(out[:, 0, 0, :4].tolist(), [[10.0, 20.0, 30.0, 10.0]])
            with self.assertRaises(NotImplementedError):
                rope(pos, [2, 1, 1], return_raw_freqs=True)

            gpt = tm.Qwen3OmniGPTModel.__new__(tm.Qwen3OmniGPTModel)
            StubGPT.__init__(gpt)
            gpt.config = SimpleNamespace(response_only_lm_head=True)
            gpt.post_process = True
            gpt.output_layer = SimpleNamespace(sequence_parallel=False)
            gpt.pg_collection = SimpleNamespace(
                cp=SimpleNamespace(size=lambda: 1),
                tp=SimpleNamespace(size=lambda: 1),
            )
            hidden = torch.randn(6, 1, 4, requires_grad=True)
            zero_mask = torch.zeros(1, 6, dtype=torch.int)
            out_logits = gpt._postprocess(hidden, loss_mask=zero_mask)
            self.assertEqual(tuple(out_logits.shape), (1, 0, 4))
            self.assertEqual(gpt.received_rows, [1])
            (out_logits.sum() + 0.0 * hidden.sum()).backward()
            self.assertIsNotNone(hidden.grad)

    def test_bug3_encoder_features_content_key_no_dataset_write_and_multistep_reuse(self):
        from slime_ext.backends.megatron.encoder_cache import EncoderFeatures

        class FakeTower(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(2), requires_grad=False)
                self.config = SimpleNamespace(attention_dropout=0, dropout=0)

        class FakeThinker(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = FakeTower()
                self.audio_model = FakeTower()
                self.image_calls = 0

            def get_image_features(self, pixel_values, grid_thw):
                self.image_calls += 1
                return pixel_values * 3.0, [pixel_values * 5.0]

            def get_audio_features(self, input_features, **kwargs):
                return input_features * 2.0

        model = FakeThinker()
        cache = EncoderFeatures([model], mode="cache", limit_gib=1.0)
        self.assertIsNone(cache.directory)

        x_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        x_b = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        grid = torch.tensor([[1, 2, 2]])

        cache.begin({"phase": "ref_log_probs", "rollout_id": 0})
        out_a_ref, _ = model.get_image_features(x_a, grid)
        out_b_ref, _ = model.get_image_features(x_b, grid)
        self.assertEqual(model.image_calls, 2)

        cache.begin({"phase": "log_probs", "rollout_id": 0})
        out_b_old, _ = model.get_image_features(x_b, grid)
        out_a_old, _ = model.get_image_features(x_a, grid)
        self.assertEqual(model.image_calls, 2)
        self.assertTrue(torch.equal(out_b_old, out_b_ref))
        self.assertTrue(torch.equal(out_a_old, out_a_ref))

        cache.begin({"phase": "actor_train", "rollout_id": 0, "step_id": 0})
        out_a_s0, _ = model.get_image_features(x_a, grid)
        cache.flush()
        cache.begin({"phase": "actor_train", "rollout_id": 0, "step_id": 1})
        out_b_s1, _ = model.get_image_features(x_b, grid)
        self.assertEqual(model.image_calls, 2)
        self.assertTrue(torch.equal(out_a_s0, out_a_ref))
        self.assertTrue(torch.equal(out_b_s1, out_b_ref))

        x_c = torch.tensor([[1.0, 2.0], [3.0, 99.0]])
        out_c, _ = model.get_image_features(x_c, grid)
        self.assertEqual(model.image_calls, 3)
        self.assertTrue(torch.equal(out_c, x_c * 3.0))

        cache.begin({"phase": "ref_log_probs", "rollout_id": 1})
        self.assertEqual(len(cache.entries), 0)
        self.assertEqual(cache.cached_bytes, 0)

    def test_bug4_rope_tail_vision_start_and_packed_multisample_visual_trimming(self):
        rope = _load_module_from_path(
            "test_rope_bug4",
            "slime_ext/models/qwen3_omni/rope.py",
        )
        IMG_TOK, VID_TOK, AUD_TOK, VIS_START, AUD_START = 151655, 151656, 151675, 151652, 151669

        ids_tail = torch.tensor([[10, 11, VIS_START, IMG_TOK, IMG_TOK, 12, VIS_START]], dtype=torch.long)
        grid_tail = torch.tensor([[1, 2, 4]], dtype=torch.long)
        pos_tail, _ = rope.get_rope_index(
            spatial_merge_size=2,
            image_token_id=IMG_TOK,
            video_token_id=VID_TOK,
            audio_token_id=AUD_TOK,
            vision_start_token_id=VIS_START,
            audio_start_token_id=AUD_START,
            position_id_per_seconds=1,
            input_ids=ids_tail,
            image_grid_thw=grid_tail,
        )
        self.assertEqual(pos_tail.shape, (3, 1, 7))

        s0 = [10, VIS_START, IMG_TOK, 99, 0, 0, 0, 0]
        s1 = [20, VIS_START, IMG_TOK, IMG_TOK, IMG_TOK, IMG_TOK, 99, 21]
        batch_ids = torch.tensor([s0, s1], dtype=torch.long)
        attn_mask = torch.tensor(
            [[True, True, True, True, False, False, False, False],
             [True, True, True, True, True, True, True, True]],
            dtype=torch.bool,
        )
        grids = torch.tensor([[1, 2, 2], [1, 2, 4], [1, 4, 4]], dtype=torch.long)
        pos, _ = rope.get_rope_index(
            spatial_merge_size=2,
            image_token_id=IMG_TOK,
            video_token_id=VID_TOK,
            audio_token_id=AUD_TOK,
            vision_start_token_id=VIS_START,
            audio_start_token_id=AUD_START,
            position_id_per_seconds=1,
            input_ids=batch_ids,
            image_grid_thw=grids,
            attention_mask=attn_mask,
            image_counts=[2, 1],
        )
        self.assertEqual(pos[1, 1, 2:6].tolist(), [2.0, 2.0, 3.0, 3.0])
        self.assertEqual(pos[2, 1, 2:6].tolist(), [2.0, 3.0, 2.0, 3.0])

        thinker_src = (REPO_ROOT / "slime_ext/models/qwen3_omni/thinker.py").read_text()
        ns = {"torch": torch}
        tree = ast.parse(thinker_src)
        fns = [
            n for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name in {"_trim_feature_sequence", "_trim_visual_features_per_sample", "_validate_deepstack_stage_layers"}
        ]
        exec(compile(ast.Module(body=fns, type_ignores=[]), "thinker_model.py", "exec"), ns)

        feats = torch.arange(7 * 2, dtype=torch.float32).reshape(7, 2)
        ds = [feats * 10.0, feats * 100.0]
        trimmed_f, trimmed_ds = ns["_trim_visual_features_per_sample"](
            feats,
            ds,
            grid_thw=grids,
            spatial_merge_size=2,
            expected_per_sample=[1, 4],
            item_counts=[2, 1],
            feature_name="Image features",
        )
        expected_rows = torch.cat([feats[0:1], feats[3:7]], dim=0)
        self.assertTrue(torch.equal(trimmed_f, expected_rows))
        self.assertTrue(torch.equal(trimmed_ds[0], expected_rows * 10.0))
        self.assertTrue(torch.equal(trimmed_ds[1], expected_rows * 100.0))

        with self.assertRaises(ValueError):
            ns["_trim_visual_features_per_sample"](
                feats, ds, grid_thw=grids, spatial_merge_size=2,
                expected_per_sample=[1, 4], item_counts=None, feature_name="Image features",
            )

        visual_stub = SimpleNamespace(deepstack_visual_indexes=[8, 16, 24])
        lm_too_small = SimpleNamespace(decoder=SimpleNamespace(layers=[object(), object()]))
        lm_ok = SimpleNamespace(decoder=SimpleNamespace(layers=[object(), object(), object()]))
        with self.assertRaises(ValueError):
            ns["_validate_deepstack_stage_layers"](True, visual_stub, lm_too_small)
        ns["_validate_deepstack_stage_layers"](True, visual_stub, lm_ok)
        ns["_validate_deepstack_stage_layers"](False, visual_stub, lm_too_small)

    def test_bug5_arguments_forbid_response_only_lm_head_with_score_centering(self):
        from slime_ext.backends.megatron.arguments import validate_omni_args

        bad = SimpleNamespace(
            response_only_lm_head=True,
            response_only_log_probs=True,
            allgather_cp=False,
            enable_mtp_training=False,
            use_score_centering=True,
            use_turn_level_gspo=False,
        )
        with self.assertRaisesRegex(ValueError, "use-score-centering"):
            validate_omni_args(bad)

        bad_gspo = SimpleNamespace(
            response_only_lm_head=False,
            use_turn_level_gspo=True,
            advantage_estimator="grpo",
        )
        with self.assertRaisesRegex(ValueError, "advantage-estimator gspo"):
            validate_omni_args(bad_gspo)

        ok_gspo = SimpleNamespace(
            response_only_lm_head=False,
            use_turn_level_gspo=True,
            advantage_estimator="gspo",
        )
        validate_omni_args(ok_gspo)

    def test_bug6_update_weight_common_tp1_zero_copy_fast_path(self):
        from slime.utils.types import ParamInfo

        from slime_ext.backends.megatron import update_weight as common

        fake_mpu = SimpleNamespace(
            get_expert_tensor_parallel_world_size=lambda: 1,
            get_tensor_model_parallel_world_size=lambda: 4,
            get_expert_tensor_parallel_group=lambda: (_ for _ in ()).throw(AssertionError("ETP=1 must not fetch group")),
            get_tensor_model_parallel_group=lambda: None,
        )

        def fail_all_gather(*args, **kwargs):
            raise AssertionError("dist.all_gather must NOT be called when tp_size == 1")

        p = torch.nn.Parameter(torch.randn(4, 4))
        p.tensor_model_parallel = True
        p.partition_dim = 0
        p.partition_stride = 1
        info = ParamInfo(
            name="module.module.thinker.language_model.decoder.layers.0.mlp.experts.linear_fc1.weight0",
            dtype=p.dtype,
            shape=p.shape,
            src_rank=0,
            size=p.numel() * p.element_size(),
            attrs={},
        )

        with patch.object(common, "mpu", fake_mpu), patch.object(common.dist, "all_gather", fail_all_gather):
            gathered = common.all_gather_param(info.name, p)
            self.assertEqual(gathered.data_ptr(), p.data_ptr())
            self.assertTrue(torch.equal(gathered, p.data))
            gathered_async = common.all_gather_params_async([(info, p)])
            self.assertEqual(len(gathered_async), 1)
            self.assertEqual(gathered_async[0].data_ptr(), p.data_ptr())
            self.assertTrue(torch.equal(gathered_async[0], p.data))

    def test_bug7_data_cpu_offsets_lazy_h2d_list_and_per_sample_num_items(self):
        from slime_ext.backends.megatron.data import _make_get_batch

        @dataclass
        class FakePackedSeqParamsWithCPUOffsets:
            cu_seqlens_q: torch.Tensor
            cu_seqlens_kv: torch.Tensor
            max_seqlen_q: int
            max_seqlen_kv: int
            qkv_format: str
            cu_seqlens_q_cpu: tuple[int, ...] | None = None

        fake_mpu = SimpleNamespace(
            get_tensor_model_parallel_world_size=lambda: 2,
            get_context_parallel_world_size=lambda: 1,
            get_context_parallel_rank=lambda: 0,
        )

        class Iter:
            def get_next(self, keys):
                return {
                    "tokens": [torch.tensor([1, 2, 3]), torch.tensor([4, 5])],
                    "loss_masks": [torch.tensor([1]), torch.tensor([1])],
                    "total_lengths": [3, 2],
                    "response_lengths": [1, 1],
                    "multimodal_train_inputs": [
                        None,
                        {"image_grid_thw": [[1, 2, 2], [1, 4, 4]], "pixel_values": torch.ones(20, 4)},
                    ],
                }

        with patch.dict(os.environ, {"SLIME_OMNI_LAZY_MULTIMODAL_H2D": "1"}):
            get_batch = _make_get_batch(fake_mpu, torch.nn.functional, FakePackedSeqParamsWithCPUOffsets, lambda t, v: t)
            batch = get_batch(Iter(), ["tokens", "loss_masks", "total_lengths", "response_lengths", "multimodal_train_inputs"], pad_multiplier=2)

        self.assertEqual(batch["packed_seq_params"].cu_seqlens_q_cpu, (0, 3, 5, 8))
        self.assertEqual(batch["packed_seq_params"].max_seqlen_q, 3)
        self.assertEqual(batch["multimodal_num_items"]["image_grid_thw"], [0, 2])
        self.assertEqual(batch["multimodal_train_inputs"]["multimodal_num_items"]["image_grid_thw"], [0, 2])

    def test_bug11_and_fp8_quantizer_migrations(self):
        from slime_ext.models.qwen3_omni.parallel import normalize_visual_outputs

        h, ds = normalize_visual_outputs((torch.ones(2, 3), [torch.zeros(2, 3), torch.ones(2, 3) * 2]))
        self.assertEqual(tuple(h.shape), (2, 3))
        self.assertEqual(len(ds), 2)

        obj = SimpleNamespace(last_hidden_state=torch.ones(2, 3), deepstack_feature_lists=[torch.zeros(2, 3)])
        _, ds2 = normalize_visual_outputs(obj)
        self.assertEqual(len(ds2), 1)

        from slime_ext.models.qwen3_omni.weights.quantizer_fp8 import quantize_qwen3omni_params_fp8

        captured = []
        fake_qmod = ModuleType("slime.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8")
        fake_qmod._quantize_param = lambda name, w, bs, ue8m0, force_ue8m0_scale=False: (
            captured.append((name, force_ue8m0_scale)) or [(name, w)]
        )
        with patch.dict(sys.modules, {"slime.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8": fake_qmod}):
            quantize_qwen3omni_params_fp8(
                SimpleNamespace(force_fp8_ue8m0_scale=True),
                "module.module.thinker.language_model.decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight",
                [("model.layers.0.mlp.experts.0.gate_proj.weight", torch.zeros(2, 2))],
                {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic", "weight_block_size": [128, 128]},
            )
        self.assertEqual(captured, [("model.layers.0.mlp.experts.0.gate_proj.weight", True)])

    def test_sglang_0521_processor_blocks_artifact_cache_and_epd_contracts(self):
        import asyncio
        from array import array

        class StubQwenVLProcessor:
            async def _process_mm_data_uncached(self, image_data, input_text, request_obj, *args, **kwargs):
                return ("uncached", input_text)

            async def process_mm_data_async(self, image_data, input_text, request_obj, *args, **kwargs):
                raise AssertionError("Artifact cache fast path should be bypassed for pretokenized input")

            async def legacy_load_mm_data(self, prompt, *args, **kwargs):
                return SimpleNamespace(input_ids=None)

        stub_mods = {}
        for m in (
            "transformers",
            "transformers.models",
            "transformers.models.qwen3_omni_moe",
            "transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe",
            "sglang",
            "sglang.srt",
            "sglang.srt.models",
            "sglang.srt.models.qwen3_omni_moe",
            "sglang.srt.multimodal",
            "sglang.srt.multimodal.processors",
            "sglang.srt.multimodal.processors.qwen_vl",
        ):
            stub_mods[m] = ModuleType(m)
        stub_mods["transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe"]._get_feat_extract_output_lengths = lambda x: x
        stub_mods["sglang.srt.models.qwen3_omni_moe"].Qwen3OmniMoeForConditionalGeneration = object
        stub_mods["sglang.srt.multimodal.processors.qwen_vl"].QwenVLImageProcessor = StubQwenVLProcessor

        with patch.dict(sys.modules, stub_mods):
            proc_mod = _load_module_from_path(
                "test_exact_token_proc_cpu", "slime_ext/backends/sglang/processor.py"
            )
            proc = proc_mod.ExactTokenQwen3OmniProcessor()
            self.assertIsNone(proc.compose_image_artifacts([1, 2, 3], []))
            res_list = asyncio.run(proc.process_mm_data_async(["img"], [10, 11], SimpleNamespace()))
            self.assertEqual(res_list, ("uncached", [10, 11]))
            res_arr = asyncio.run(proc.process_mm_data_async(["img"], array("q", [10, 11]), SimpleNamespace()))
            self.assertEqual(res_arr[0], "uncached")
            legacy = asyncio.run(proc.legacy_load_mm_data(array("q", [10, 11])))
            self.assertEqual(legacy.input_ids, [10, 11])

        se_tree = ast.parse((SLIME_ROOT / "slime/backends/sglang_utils/sglang_engine.py").read_text())
        compute_args = None
        for node in ast.walk(se_tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_compute_server_args":
                compute_args = [a.arg for a in node.args.args]
        self.assertIsNotNone(compute_args)
        self.assertIn("encoder_bootstrap_port", compute_args)


class P0P1RefactorRegressionTests(unittest.TestCase):
    def test_megatron_deepep_opt_in_and_rollout_id_advances_without_kl_loss(self):
        from slime_ext.backends import megatron as ext_megatron

        calls = []
        fake_fused_a2a = ModuleType("megatron.core.transformer.moe.fused_a2a")
        fake_fused_a2a.set_deepep_num_sms = lambda n: calls.append(("set", n))
        fake_fused_a2a.get_buffer = lambda group, size: calls.append(("buf", group, size))
        fake_mods = {
            "megatron": ModuleType("megatron"),
            "megatron.core": ModuleType("megatron.core"),
            "megatron.core.transformer": ModuleType("megatron.core.transformer"),
            "megatron.core.transformer.moe": ModuleType("megatron.core.transformer.moe"),
            "megatron.core.transformer.moe.fused_a2a": fake_fused_a2a,
        }

        class FakeMoEPart(torch.nn.Module):
            def __init__(self, with_manager: bool):
                super().__init__()
                if with_manager:
                    self.token_dispatcher = SimpleNamespace(_comm_manager=SimpleNamespace(group="ep_group"))

        args = SimpleNamespace(hidden_size=64, start_rollout_id=0)
        with patch.dict(sys.modules, fake_mods):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("SLIME_OMNI_DEEPEP_NUM_SMS", None)
                ext_megatron._buffer_reserved = False
                ext_megatron._reserve_deepep(args, [FakeMoEPart(True)])
                self.assertEqual(calls, [])

            with patch.dict(os.environ, {"SLIME_OMNI_DEEPEP_NUM_SMS": "0"}):
                ext_megatron._buffer_reserved = False
                ext_megatron._reserve_deepep(args, [FakeMoEPart(True)])
                self.assertEqual(calls, [])

            with patch.dict(os.environ, {"SLIME_OMNI_DEEPEP_NUM_SMS": "24"}):
                ext_megatron._buffer_reserved = False
                ext_megatron._reserve_deepep(args, [FakeMoEPart(False)])
                self.assertEqual(calls, [])

                ext_megatron._buffer_reserved = False
                ext_megatron._reserve_deepep(args, [FakeMoEPart(True)])
                self.assertEqual(calls, [("set", 24), ("buf", "ep_group", 128)])

        seen_contexts = []
        fake_cache = SimpleNamespace(begin=lambda ctx: seen_contexts.append(dict(ctx)), flush=lambda: None)
        fake_opt = SimpleNamespace(prepare_grads=lambda: None)
        with patch.object(ext_megatron, "_encoder_features", fake_cache), patch.object(
            ext_megatron, "_reserve_deepep", lambda a, m: None
        ), patch.object(ext_megatron.cpu_mrope, "install", lambda: None), patch.object(
            ext_megatron, "_rollout_id", -1
        ), patch.object(ext_megatron, "_in_train", False), patch.object(
            ext_megatron, "_flush_installed", False
        ), patch.dict(os.environ, {"SLIME_OMNI_ENCODER_FEATURE_MODE": "cache"}):
            ext_megatron._seen_log_prob_prefixes.clear()
            ext_megatron.before_log_prob(args, [], store_prefix="")
            ext_megatron.before_train(args, rollout_id=0, step_id=0, model=[], optimizer=fake_opt, scheduler=None)
            ext_megatron.before_log_prob(args, [], store_prefix="")
            ext_megatron.before_train(args, rollout_id=1, step_id=0, model=[], optimizer=fake_opt, scheduler=None)
            ext_megatron.before_log_prob(args, [], store_prefix="ref_")
            ext_megatron.before_log_prob(args, [], store_prefix="")
            ext_megatron.before_train(args, rollout_id=2, step_id=0, model=[], optimizer=fake_opt, scheduler=None)
            ext_megatron.before_log_prob(args, [], store_prefix="")
            ext_megatron.before_log_prob(args, [], store_prefix="")
            ext_megatron._seen_log_prob_prefixes.clear()

        self.assertEqual(
            [(c["phase"], c["rollout_id"]) for c in seen_contexts],
            [
                ("log_probs", 0),
                ("actor_train", 0),
                ("log_probs", 1),
                ("actor_train", 1),
                ("ref_log_probs", 2),
                ("log_probs", 2),
                ("actor_train", 2),
                ("log_probs", 3),
                ("log_probs", 4),
            ],
        )

    def test_encoder_cache_no_cuda_events_without_dir_and_o1_fingerprint(self):
        from slime_ext.backends.megatron import encoder_cache as ec

        class FakeTower(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(2), requires_grad=False)
                self.config = SimpleNamespace(attention_dropout=0, dropout=0)

        class FakeThinker(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = FakeTower()
                self.audio_model = FakeTower()
                self.calls = 0

            def get_image_features(self, pixel_values, grid_thw):
                self.calls += 1
                return pixel_values * 2.0, [pixel_values * 4.0]

            def get_audio_features(self, input_features, **kwargs):
                return input_features

        def fail_cuda_event(*args, **kwargs):
            raise AssertionError("CUDA Event must not be allocated when directory is None")

        def fail_cuda_sync(*args, **kwargs):
            raise AssertionError("torch.cuda.synchronize must not be called when directory is None")

        model = FakeThinker()
        cache = ec.EncoderFeatures([model], mode="cache", limit_gib=0.5)
        self.assertIsNone(cache.directory)
        with patch.object(torch.cuda, "Event", fail_cuda_event), patch.object(
            torch.cuda, "synchronize", fail_cuda_sync
        ):
            cache.begin({"phase": "log_probs", "rollout_id": 0})
            x = torch.randn(4, 4)
            grid = torch.tensor([[1, 2, 2]])
            out1, _ = model.get_image_features(x, grid)
            out2, _ = model.get_image_features(x, grid)
            cache.flush()

        self.assertEqual(model.calls, 1)
        self.assertTrue(torch.equal(out1, out2))
        self.assertEqual(cache.records, [])

        def fail_arange(*args, **kwargs):
            raise AssertionError("torch.arange must not be allocated in _tensor_fingerprint")

        torch.manual_seed(7)
        for dtype in (torch.float32, torch.bfloat16, torch.int64, torch.bool):
            if dtype == torch.bool:
                big = torch.randint(0, 2, (128, 64), dtype=torch.bool)
            elif dtype == torch.int64:
                big = torch.randint(-1000, 1000, (128, 64), dtype=torch.int64)
            else:
                big = torch.randn(128, 64, dtype=dtype)
            with patch.object(ec.torch, "arange", fail_arange):
                fp_a = ec._tensor_fingerprint(big)
                fp_same = ec._tensor_fingerprint(big.clone())
                self.assertEqual(fp_a, fp_same)

                edited = big.clone()
                if dtype == torch.bool:
                    edited.view(-1)[1] = not bool(edited.view(-1)[1].item())
                else:
                    edited.view(-1)[1] = edited.view(-1)[1] + 1
                self.assertNotEqual(fp_a, ec._tensor_fingerprint(edited))

                swapped = big.clone()
                flat_s = swapped.view(-1)
                j = 2
                while j < flat_s.numel() and flat_s[j].item() == flat_s[0].item():
                    j += 2
                if j < flat_s.numel():
                    v0, vj = flat_s[0].item(), flat_s[j].item()
                    flat_s[0], flat_s[j] = vj, v0
                    self.assertNotEqual(fp_a, ec._tensor_fingerprint(swapped))

    def test_actor_lazy_multimodal_h2d_wraps_original_and_scaffolding_deleted(self):
        fake_megatron = ModuleType("megatron")
        fake_megatron_core = ModuleType("megatron.core")
        fake_megatron_core.mpu = SimpleNamespace()
        fake_cp_utils = ModuleType("slime.backends.megatron_utils.cp_utils")
        fake_cp_utils.prepare_routed_experts_for_routing_replay = lambda *a, **kw: None
        fake_mod = ModuleType("slime.backends.megatron_utils.actor")

        def fake_process_rollout_data(args, data_id, data_partitions):
            return dict(data_partitions[0])

        fake_mod.process_rollout_data = fake_process_rollout_data
        fake_mod.accelerator = SimpleNamespace(current_device=lambda: torch.device("cpu"))
        fake_mod.slice_log_prob_with_cp = lambda t, t_len, r_len: t

        stub_mods = {
            "megatron": fake_megatron,
            "megatron.core": fake_megatron_core,
            "slime.backends.megatron_utils.cp_utils": fake_cp_utils,
            "slime.backends.megatron_utils.actor": fake_mod,
        }
        with patch.dict(sys.modules, stub_mods):
            ext_actor = _load_module_from_path(
                "test_slime_ext_actor_cpu", "slime_ext/backends/megatron/actor.py"
            )

            for deleted in (
                "_debug_routing_replay_enabled",
                "_routing_boundary_debug_path",
                "_dump_routing_boundary_record",
                "_dump_raw_routing_boundary",
            ):
                self.assertFalse(hasattr(ext_actor, deleted), f"{deleted} should be removed from actor.py")

            seen_in_upstream = []

            def fake_original(actor_self, rollout_data_ref):
                processed = fake_mod.process_rollout_data(actor_self.args, 0, [rollout_data_ref])
                seen_in_upstream.append(dict(processed))
                return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in processed.items()}

            mm_cpu = [{"pixel_values": torch.ones(2, 3)}]
            raw_ref = {
                "tokens": [torch.tensor([1, 2, 3])],
                "total_lengths": [3],
                "response_lengths": [2],
                "multimodal_train_inputs": mm_cpu,
                "token_advantages": [[0.5, -0.5]],
                "turn_ids": [[0, 1]],
                "turn_weights": [[0.5, 0.5]],
            }
            wrapped = ext_actor._make_get_rollout_data(fake_original)
            actor_stub = SimpleNamespace(args=SimpleNamespace())

            with patch.dict(os.environ, {"SLIME_OMNI_LAZY_MULTIMODAL_H2D": "1"}):
                out = wrapped(actor_stub, raw_ref)

        self.assertNotIn("multimodal_train_inputs", seen_in_upstream[0])
        self.assertIs(out["multimodal_train_inputs"], mm_cpu)
        self.assertEqual(out["token_advantages"][0].dtype, torch.float32)
        self.assertEqual(out["turn_ids"][0].dtype, torch.long)
        self.assertEqual(out["turn_weights"][0].dtype, torch.float32)
        self.assertIs(fake_mod.process_rollout_data, fake_process_rollout_data)

    def test_loss_split_explicit_projection_and_unused_cli_flags_removed(self):
        import argparse

        from slime_ext.backends.megatron import (
            arguments as ext_args,
        )
        from slime_ext.backends.megatron import (
            loss as ext_loss,
        )
        from slime_ext.backends.megatron import (
            response_only_loss,
            turn_level_gspo,
        )

        self.assertFalse(hasattr(ext_loss, "ExplicitResponseProjection"))
        self.assertFalse(hasattr(response_only_loss, "ExplicitResponseProjection"))
        text_src = (REPO_ROOT / "slime_ext/models/qwen3_omni/text.py").read_text(encoding="utf-8")
        self.assertNotIn("ExplicitResponseProjection", text_src)

        self.assertIs(ext_loss.get_log_probs_and_entropy, response_only_loss.get_log_probs_and_entropy)
        self.assertIs(ext_loss.compute_turn_level_gspo_loss, turn_level_gspo.compute_turn_level_gspo_loss)
        self.assertIs(ext_loss.policy_loss_function, turn_level_gspo.policy_loss_function)

        parser = argparse.ArgumentParser()
        ext_args._with_slime_omni_arguments(None)(parser)
        option_strings = {opt for action in parser._actions for opt in action.option_strings}
        self.assertNotIn("--rollout-data-prefetch-depth", option_strings)
        self.assertNotIn("--custom-dp-schedule-function-path", option_strings)
        self.assertIn("--use-turn-level-gspo", option_strings)
        self.assertIn("--response-only-log-probs", option_strings)

    def test_qwen3_omni_and_sglang_top_p_cleanups(self):
        from slime_ext.models.qwen3_omni.weights.naming import strip_module_prefixes

        self.assertEqual(strip_module_prefixes("module.module.thinker.weight"), "thinker.weight")
        self.assertEqual(strip_module_prefixes("thinker.weight"), "thinker.weight")

        rope = _load_module_from_path("test_rope_cleanup", "slime_ext/models/qwen3_omni/rope.py")
        self.assertFalse(hasattr(rope, "get_rope_index_cpu_arrays"))

        thinker_src = (REPO_ROOT / "slime_ext/models/qwen3_omni/thinker.py").read_text(encoding="utf-8")
        self.assertIn("def _encode_and_trim_visual(", thinker_src)
        self.assertIn("def _merge_multimodal_features(", thinker_src)
        for dead in (
            "multimodal_row_merge",
            "talker_transformer_config",
            "code2wav_transformer_config",
            "seconds_per_chunk",
            "encoder_hidden_state",
            "add_encoder",
        ):
            self.assertNotIn(dead, thinker_src)

        from slime_ext.backends.sglang import top_p as ext_top_p

        def _pkg(name):
            mod = ModuleType(name)
            mod.__path__ = []
            return mod

        fake_sglang = {
            "sglang": _pkg("sglang"),
            "sglang.srt": _pkg("sglang.srt"),
            "sglang.srt.layers": _pkg("sglang.srt.layers"),
            "sglang.srt.layers.utils": _pkg("sglang.srt.layers.utils"),
            "sglang.srt.layers.sampler": ModuleType("sglang.srt.layers.sampler"),
            "sglang.srt.layers.utils.top_p": ModuleType("sglang.srt.layers.utils.top_p"),
            "sglang.srt.layers.logprob_processor": ModuleType("sglang.srt.layers.logprob_processor"),
        }
        fake_sglang["sglang.srt.layers"].sampler = fake_sglang["sglang.srt.layers.sampler"]
        fake_sglang["sglang.srt.layers"].utils = fake_sglang["sglang.srt.layers.utils"]
        fake_sglang["sglang.srt.layers"].logprob_processor = fake_sglang["sglang.srt.layers.logprob_processor"]
        fake_sglang["sglang.srt.layers.utils"].top_p = fake_sglang["sglang.srt.layers.utils.top_p"]

        sentinel_filter = lambda *a, **kw: None
        sentinel_mask = lambda *a, **kw: None
        fake_sglang["sglang.srt.layers.utils.top_p"]._top_p_filter_rows = sentinel_filter
        fake_sglang["sglang.srt.layers.utils.top_p"]._top_p_keep_mask_sorted = sentinel_mask
        fake_sglang["sglang.srt.layers.utils.top_p"].get_top_p_token_ids_from_probs = lambda *a, **kw: None
        fake_sglang["sglang.srt.layers.sampler"].get_top_p_token_ids_from_probs = lambda *a, **kw: None
        fake_sglang["sglang.srt.layers.logprob_processor"].get_top_p_token_ids_from_probs = lambda *a, **kw: None

        with patch.dict(sys.modules, fake_sglang), patch.object(ext_top_p, "_applied", False), patch.object(
            ext_top_p, "_top_p_filter_rows", None
        ), patch.object(ext_top_p, "_top_p_keep_mask_sorted", None):
            ext_top_p.patch()
            self.assertIs(
                fake_sglang["sglang.srt.layers.utils.top_p"].get_top_p_token_ids_from_probs,
                ext_top_p.get_top_p_token_ids_from_probs,
            )
            self.assertIs(
                fake_sglang["sglang.srt.layers.sampler"].get_top_p_token_ids_from_probs,
                ext_top_p.get_top_p_token_ids_from_probs,
            )
            self.assertIs(
                fake_sglang["sglang.srt.layers.logprob_processor"].get_top_p_token_ids_from_probs,
                ext_top_p.get_top_p_token_ids_from_probs,
            )
            self.assertIs(ext_top_p._top_p_filter_rows, sentinel_filter)
            self.assertIs(ext_top_p._top_p_keep_mask_sorted, sentinel_mask)

            marker_fn = lambda *a, **kw: "custom"
            fake_sglang["sglang.srt.layers.sampler"].get_top_p_token_ids_from_probs = marker_fn
            ext_top_p.patch()
            self.assertIs(fake_sglang["sglang.srt.layers.sampler"].get_top_p_token_ids_from_probs, marker_fn)

    def test_train_entrypoint_optional_ray_init(self):
        from slime_ext import train as ext_train

        argv = [
            "--ray-address",
            "127.0.0.1:6379",
            "--ray-env-vars-json",
            '{"FOO": "bar"}',
            "--advantage-estimator",
            "gspo",
        ]
        opts, rest = ext_train._extract_ray_options(argv)
        self.assertEqual(opts["ray_address"], "127.0.0.1:6379")
        self.assertEqual(opts["ray_env_vars_json"], '{"FOO": "bar"}')
        self.assertEqual(rest, ["--advantage-estimator", "gspo"])

        init_calls = []
        fake_ray = ModuleType("ray")
        fake_ray.is_initialized = lambda: bool(init_calls)
        fake_ray.init = lambda **kw: init_calls.append(kw)

        with patch.dict(sys.modules, {"ray": fake_ray}), patch.dict(os.environ, {}, clear=False):
            for k in ("SLIME_RAY_ADDRESS", "SLIME_RAY_RUNTIME_ENV_JSON", "SLIME_RAY_ENV_VARS_JSON"):
                os.environ.pop(k, None)
            rem_plain = ext_train._maybe_init_ray(["--advantage-estimator", "gspo"])
            self.assertEqual(rem_plain, ["--advantage-estimator", "gspo"])
            self.assertEqual(init_calls, [])

            rem_ray = ext_train._maybe_init_ray(argv)
            self.assertEqual(rem_ray, ["--advantage-estimator", "gspo"])
            self.assertEqual(
                init_calls,
                [{"address": "127.0.0.1:6379", "runtime_env": {"env_vars": {"FOO": "bar"}}}],
            )
            ext_train._maybe_init_ray(argv)
            self.assertEqual(len(init_calls), 1)


if __name__ == "__main__":
    unittest.main()
