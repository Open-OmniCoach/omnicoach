"""CPU contract tests for shared slime_ext.agent & Example 2 (qwen3_omni_agent_gspo)."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

from examples.qwen3_omni_agent_gspo.configure import resolve
from examples.qwen3_omni_agent_gspo.data import prepare_jsonl
from examples.qwen3_omni_agent_gspo.reward import (
    convert_samples_to_train_data,
    filter_group,
    log_metrics,
    score,
)
from examples.qwen3_omni_agent_gspo.rollout import generate
from slime_ext.agent.media import Media
from slime_ext.agent.protocol import (
    DEFAULT_MEDIA_SETTINGS,
    AgentSession,
    ProtocolError,
    parse_tool_call,
)
from slime_ext.agent.rewards import (
    filter_outcome_group,
    log_agent_metrics,
    reject_runtime_failures,
)
from slime_ext.agent.rollout import (
    AgentSample,
    ContextLengthExceeded,
    FallbackStatus,
    apply_turn_generation_result,
    safe_min_weight_version,
    safe_policy_lag,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = ROOT / "examples" / "qwen3_omni_agent_gspo"

FORBIDDEN_DEPRECATED_TOKENS = (
    "train_async",
    "--update-weights-interval",
    "--keep-old-actor",
    "--update-weight-load-format",
    "--disable-rollout-global-dataset",
    "--rollout-max-policy-lag",
    "--rollout-max-prefetch-groups",
    "--compact-rollout-routing",
    "--overlap-weight-sync-with-rollout",
)


def _agent_args(**overrides):
    cfg = resolve(EXAMPLE_DIR / "config.yaml", {})
    base = {
        "agent_config": cfg["agent_config"],
        "n_samples_per_prompt": 2,
        "rewards_normalization": True,
        "grpo_std_normalization": False,
        "use_rollout_logprobs": True,
        "rollout_top_p": 0.95,
        "use_rollout_routing_replay": False,
        "partial_rollout": True,
        "mask_offpolicy_in_partial_rollout": True,
        "rollout_num_gpus": 2,
    }
    base.update(overrides)
    return types.SimpleNamespace(**base)


class TestAgentGSPOContract(unittest.TestCase):
    # --- Trailing-window clamp & short-video (<1s) guard ---
    def test_trailing_clamp_and_short_video_guard(self) -> None:
        # Video duration = 10.2s (ceil=11): [10.5s, 11s] must be REJECTED (not end < start!)
        bad_call = {
            "id": "c0",
            "function": {"name": "get_audio", "arguments": {"start": "10.5s", "end": "11s"}},
        }
        with self.assertRaises(ProtocolError) as ctx:
            parse_tool_call(bad_call, duration=10.2, has_audio=True, settings=DEFAULT_MEDIA_SETTINGS)
        self.assertEqual(ctx.exception.code, "INVALID_MEDIA_RANGE")

        # Multi-span negative-duration smuggling must also be rejected
        bad_multi = {
            "id": "c1",
            "function": {
                "name": "get_multi_span",
                "arguments": {
                    "spans": [
                        {"modality": "audio", "start": "0s", "end": "128s"},
                        {"modality": "audio", "start": "200.9s", "end": "201s"},
                    ]
                },
            },
        }
        with self.assertRaises(ProtocolError) as ctx2:
            parse_tool_call(bad_multi, duration=200.2, has_audio=True, settings=DEFAULT_MEDIA_SETTINGS)
        self.assertEqual(ctx2.exception.code, "INVALID_MEDIA_RANGE")

        # Short video (< 1s): duration=0.6s supports valid sub-second calls & Media slicing
        short_media = Media.synthetic(duration_s=0.6, has_audio=True, sample_fps=4.0)
        try:
            init_bundle, attached, media_state = short_media.initial()
            self.assertTrue(attached)
            self.assertGreater(init_bundle.frame_count, 0)
            self.assertAlmostEqual(media_state.duration_s, 0.6, places=3)

            valid_short = {
                "id": "c2",
                "function": {
                    "name": "get_clip",
                    "arguments": {"start": "0.1s", "end": "0.5s", "level": "detail"},
                },
            }
            req = parse_tool_call(valid_short, duration=0.6, has_audio=True, settings=DEFAULT_MEDIA_SETTINGS)
            self.assertGreater(req.end, req.start)
            self.assertLessEqual(req.end, 0.6)
            bundle = short_media.execute(req)
            self.assertGreater(bundle.frame_count, 0)
            self.assertGreater(bundle.cost_units, 0.0)
        finally:
            short_media.close()

    # --- Canonical duplicate identity ---
    def test_canonical_duplicate_identity(self) -> None:
        c1 = {
            "id": "c1",
            "function": {"name": "get_frames", "arguments": {"start": "10s", "end": "20s", "level": "coarse"}},
        }
        c2 = {
            "id": "c2",
            "function": {
                "name": "get_frames",
                "arguments": {"start": "10.0s", "end": "20.0s", "level": "coarse", "resolution": "default"},
            },
        }
        r1 = parse_tool_call(c1, duration=60.0, has_audio=True, settings=DEFAULT_MEDIA_SETTINGS)
        r2 = parse_tool_call(c2, duration=60.0, has_audio=True, settings=DEFAULT_MEDIA_SETTINGS)
        self.assertEqual(r1.identity, r2.identity)

        # Reversed get_multi_span order remains distinct
        m1 = parse_tool_call(
            {
                "id": "m1",
                "function": {
                    "name": "get_multi_span",
                    "arguments": {
                        "spans": [
                            {"modality": "frames", "start": "5s", "end": "10s", "level": "coarse"},
                            {"modality": "frames", "start": "20s", "end": "25s", "level": "coarse"},
                        ]
                    },
                },
            },
            duration=60.0,
            has_audio=True,
            settings=DEFAULT_MEDIA_SETTINGS,
        )
        m2 = parse_tool_call(
            {
                "id": "m2",
                "function": {
                    "name": "get_multi_span",
                    "arguments": {
                        "spans": [
                            {"modality": "frames", "start": "20s", "end": "25s", "level": "coarse"},
                            {"modality": "frames", "start": "5s", "end": "10s", "level": "coarse"},
                        ]
                    },
                },
            },
            duration=60.0,
            has_audio=True,
            settings=DEFAULT_MEDIA_SETTINGS,
        )
        self.assertNotEqual(m1.identity, m2.identity)

    # --- [NOTICE] does not truncate assistant or user question text ---
    def test_notice_never_truncates_assistant_or_user_body(self) -> None:
        media = Media.synthetic(duration_s=10.0, has_audio=True)
        try:
            init_b, attached, _ = media.initial()
            session = AgentSession.create(
                prompt="Question mentions [NOTICE] in the prompt body. Choose A or B.",
                duration_seconds=10.0,
                has_audio=True,
                initial_audio_attached=attached,
                initial=init_b,
                settings=DEFAULT_MEDIA_SETTINGS,
                frame_contract=media.frame_contract,
            )
            session.append_control_state(step=0, max_steps=8)
            session.append_assistant(
                {"role": "assistant", "content": "<observation>Saw [NOTICE] sign on wall.</observation>\n<think>ok</think>"}
            )
            session.append_control_state(step=1, max_steps=8)
            # User prompt body preserved intact
            user_text = session.messages[1]["content"][0]["text"]
            self.assertIn("Question mentions [NOTICE] in the prompt body.", user_text)
            # Assistant message preserved intact
            self.assertIn("Saw [NOTICE] sign on wall.", session.messages[2]["content"])
        finally:
            media.close()

    # --- Partial rollout resume, off-policy mask, empty-turn overflow, 0-token abort ---
    def test_partial_rollout_resume_offpolicy_mask_and_runtime_guards(self) -> None:
        args = _agent_args(partial_rollout=True, mask_offpolicy_in_partial_rollout=True)

        # 1) Multi-turn episode with mid-turn pause & resume across weight versions
        call_counter = {"n": 0}

        async def mock_gen(a, turn, episode, sampling):
            idx = call_counter["n"]
            call_counter["n"] += 1
            if idx == 0:
                # Turn 0: tool call generated under weight_version=1
                apply_turn_generation_result(
                    a,
                    turn,
                    new_tokens=[10, 11, 12, 13],
                    new_log_probs=[-0.2, -0.3, -0.1, -0.2],
                    text=(
                        "<observation>Global frames checked.</observation>\n"
                        "<think>Inspect 2s-5s clip.</think>\n"
                        '<tool_call>\n{"name": "get_clip", "arguments": {"start": "2s", "end": "5s", "level": "detail"}}\n</tool_call>'
                    ),
                    finish_reason="stop",
                    weight_version=1,
                )
            elif idx == 1:
                # Turn 1 attempt 1: 0-token abort must NOT append weight_version=1
                apply_turn_generation_result(
                    a,
                    turn,
                    new_tokens=[],
                    new_log_probs=[],
                    text="",
                    finish_reason="abort",
                    weight_version=1,
                )
            elif idx == 2:
                # Turn 1 attempt 2: partial generation (2 tokens under v=2), then paused!
                apply_turn_generation_result(
                    a,
                    turn,
                    new_tokens=[20, 21],
                    new_log_probs=[-0.4, -0.5],
                    text="<observation>Detail clip ",
                    finish_reason="abort",
                    weight_version=2,
                )
            else:
                # Turn 1 resumed under v=3: completes answer (3 new tokens)
                apply_turn_generation_result(
                    a,
                    turn,
                    new_tokens=[22, 23, 24],
                    new_log_probs=[-0.1, -0.1, -0.1],
                    text="shows B.</observation>\n<think>Confirmed.</think>\n<answer>B</answer>",
                    finish_reason="stop",
                    weight_version=3,
                )

        sample = AgentSample(
            index=0,
            group_index=0,
            rollout_id=0,
            prompt="Which option?",
            label="B",
            metadata={"task_id": "t1", "answer_type": "mcq", "source_name": "unit"},
        )
        media_factory = lambda _s, _cfg: Media.synthetic(duration_s=10.0, has_audio=True)

        # Pause 1 (0-token abort on turn 1)
        paused1 = asyncio.run(
            generate(args, sample, {"max_new_tokens": 64}, media_factory=media_factory, turn_generator=mock_gen)
        )
        self.assertEqual(paused1.status, FallbackStatus.ABORTED)
        # Turn 1 active had 0 tokens -> weight_versions must be empty
        self.assertEqual(paused1.metadata["episode"].active.weight_versions, [])
        # Neutral initial_media_state persisted across pause with recipe-agnostic keys
        saved_init_state = paused1.metadata["initial_media_state"]
        self.assertTrue(saved_init_state["initial_observations"])
        self.assertTrue(saved_init_state["covered_spans"]["av"])
        self.assertNotIn("initial_candidates", saved_init_state)
        self.assertNotIn("AV", saved_init_state["covered_spans"])

        # Pause 2 (2 partial tokens on turn 1)
        paused2 = asyncio.run(
            generate(args, paused1, {"max_new_tokens": 64}, media_factory=media_factory, turn_generator=mock_gen)
        )
        self.assertEqual(paused2.status, FallbackStatus.ABORTED)
        self.assertEqual(paused2.metadata["episode"].active.weight_versions, [2])

        # Resume & Finish under v=3
        finished = asyncio.run(
            generate(args, paused2, {"max_new_tokens": 64}, media_factory=media_factory, turn_generator=mock_gen)
        )
        self.assertEqual(finished.status, FallbackStatus.COMPLETED)
        self.assertEqual(finished.metadata["answer"], "B")
        self.assertEqual(finished.metadata["turn_roles"], ["tool", "answer"])
        self.assertEqual(finished.metadata["turn_spans"], [(0, 4), (4, 9)])
        self.assertEqual(finished.metadata["turn_weight_versions"], [1, 2])
        # Initial media state & neutral tool observations preserved cleanly across pauses
        self.assertEqual(
            finished.metadata["initial_media_state"]["initial_observations"],
            saved_init_state["initial_observations"],
        )
        self.assertEqual(
            finished.metadata["initial_observations"],
            saved_init_state["initial_observations"],
        )
        agent_turns = finished.metadata["agent_turns"]
        self.assertEqual(len(agent_turns), 2)
        self.assertEqual(agent_turns[0]["role"], "tool")
        self.assertEqual(agent_turns[1]["role"], "answer")
        tool_obs = agent_turns[0]["tool_observations"]
        self.assertEqual(len(tool_obs), 1)
        self.assertEqual(
            set(tool_obs[0].keys()),
            {"tool_name", "media_kind", "observed_spans", "frame_timestamps", "duration_sec", "params"},
        )
        self.assertEqual(tool_obs[0]["tool_name"], "get_clip")
        self.assertEqual(tool_obs[0]["media_kind"], "av")
        self.assertEqual(tool_obs[0]["observed_spans"], [(2.0, 5.0)])
        self.assertEqual(tool_obs[0]["params"]["turn_index"], 0)
        for rec in list(finished.metadata["initial_observations"]) + list(tool_obs):
            for forbidden in ("fidelity", "candidate_id", "support_spans"):
                self.assertNotIn(forbidden, rec)
                self.assertNotIn(forbidden, rec["params"])
        # Off-policy masking: past turn 0 masked [0,0,0,0], turn 1 had first 2 tokens masked [0,0] and last 3 tokens [1,1,1]
        self.assertEqual(finished.metadata["turns"][0].loss_mask, [0, 0, 0, 0])
        self.assertEqual(finished.metadata["turns"][1].loss_mask, [0, 0, 1, 1, 1])

        # 2) Turn-0 ContextLengthExceeded (empty turns=[]) does not crash or livelock
        async def overflow_gen(_a, _t, _ep, _sp):
            raise ContextLengthExceeded("too long")

        overflow_sample = AgentSample(
            index=1,
            group_index=1,
            rollout_id=1,
            prompt="Long prompt",
            label="A",
            metadata={"task_id": "t2", "answer_type": "mcq", "source_name": "unit"},
        )
        failed = asyncio.run(
            generate(args, overflow_sample, {"max_new_tokens": 64}, media_factory=media_factory, turn_generator=overflow_gen)
        )
        self.assertEqual(failed.status, FallbackStatus.FAILED)
        self.assertEqual(failed.metadata["turns"], [])
        self.assertEqual(failed.metadata["terminal_status"], "FAILED")
        self.assertEqual(failed.metadata["failure_reason"], "CONTEXT_LENGTH_EXCEEDED")
        # Safe lag helpers never raise on empty versions or None training_version
        self.assertIsNone(safe_min_weight_version([failed]))
        self.assertIsNone(safe_policy_lag(5, [failed]))
        self.assertIsNone(safe_policy_lag(None, [finished]))
        # Re-entry restores FAILED immediately (no ABORTED livelock!)
        failed.status = FallbackStatus.ABORTED
        reentered = asyncio.run(
            generate(args, failed, {"max_new_tokens": 64}, media_factory=media_factory, turn_generator=overflow_gen)
        )
        self.assertEqual(reentered.status, FallbackStatus.FAILED)
        # Runtime failure filter rejects overflow cleanly
        self.assertFalse(filter_group(args, [failed, finished]).keep)

        # 3) MEDIA_TOOL_FAILED rejected & does not pollute turns
        class FailingMedia(Media):
            def execute(self, request, *, _record_budget=True):
                raise RuntimeError("DECORDError simulated")

        def failing_factory(_s, _cfg):
            m = Media.synthetic(duration_s=10.0, has_audio=True)
            m.__class__ = FailingMedia
            return m

        async def tool_call_gen(a, turn, _ep, _sp):
            apply_turn_generation_result(
                a,
                turn,
                new_tokens=[1, 2],
                new_log_probs=[-0.1, -0.1],
                text=(
                    "<observation>Init.</observation>\n<think>Clip.</think>\n"
                    '<tool_call>\n{"name": "get_clip", "arguments": {"start": "1s", "end": "3s", "level": "coarse"}}\n</tool_call>'
                ),
                finish_reason="stop",
                weight_version=1,
            )

        bad_media_sample = AgentSample(
            index=2,
            group_index=2,
            rollout_id=2,
            prompt="Bad video",
            label="A",
            metadata={"task_id": "t3", "answer_type": "mcq", "source_name": "unit"},
        )
        media_failed = asyncio.run(
            generate(args, bad_media_sample, {"max_new_tokens": 64}, media_factory=failing_factory, turn_generator=tool_call_gen)
        )
        self.assertEqual(media_failed.metadata["failure_reason"], "MEDIA_TOOL_FAILED")
        self.assertEqual(media_failed.metadata["turns"], [])
        self.assertFalse(reject_runtime_failures(args, [media_failed]).keep)

        # 4) 0-token stop initializes valid empty top-p offsets
        zero_turn = AgentSample(index=9, group_index=0, rollout_id=9, tokens=[1, 2])
        apply_turn_generation_result(
            args, zero_turn, new_tokens=[], new_log_probs=[], text="", finish_reason="stop", weight_version=1
        )
        self.assertEqual(zero_turn.rollout_top_p_token_ids, [])
        self.assertEqual(zero_turn.rollout_top_p_token_offsets, [0])

    def test_pure_outcome_gspo_reward_conversion_and_metrics(self) -> None:
        args = _agent_args(n_samples_per_prompt=2, partial_rollout=False, mask_offpolicy_in_partial_rollout=False)

        def _make_done(idx: int, ans: str, turns_n: int):
            turns = [
                AgentSample(
                    index=idx,
                    group_index=0,
                    rollout_id=idx,
                    tokens=[1, 2, 3, 4],
                    response_length=2,
                    loss_mask=[1, 1],
                    rollout_log_probs=[-0.2, -0.3],
                    rollout_top_p_token_ids=[1, 2],
                    rollout_top_p_token_offsets=[0, 1, 2],
                    status=FallbackStatus.COMPLETED,
                    metadata={"source_name": "unit", "turn_index": k},
                )
                for k in range(turns_n)
            ]
            return AgentSample(
                index=idx,
                group_index=0,
                rollout_id=idx,
                label="A",
                status=FallbackStatus.COMPLETED,
                metadata={
                    "task_id": "t_gspo",
                    "answer_type": "mcq",
                    "source_name": "unit",
                    "answer": ans,
                    "invalid_actions": 0,
                    "no_progress": 0,
                    "turns": turns,
                },
            )

        s_good = _make_done(0, "A", 1)
        s_bad = _make_done(1, "B", 3)
        self.assertEqual(asyncio.run(score(args, s_good)), 1.0)
        self.assertEqual(asyncio.run(score(args, s_bad)), 0.0)
        self.assertTrue(filter_outcome_group(args, [s_good, s_bad]).keep)

        train_data = convert_samples_to_train_data(args, [s_good, s_bad])
        self.assertEqual(len(train_data["tokens"]), 4)  # 1 + 3 turns
        # Pure outcome centering: +0.5 on s_good turn, -0.5 on all 3 s_bad turns
        self.assertEqual(train_data["rewards"], [0.5, -0.5, -0.5, -0.5])
        self.assertEqual(train_data["rollout_mask_sums"], [2, 6, 6, 6])
        self.assertNotIn("token_advantages", train_data)
        self.assertNotIn("turn_ids", train_data)

        # log_metrics handles None safely & reports unbiased candidate mean (0.5, not 0.25)
        self.assertFalse(log_metrics(0, args, [s_good, s_bad], None, 1.0))
        metrics = {}
        log_agent_metrics(0, args, [s_good, s_bad], metrics, 2.0)
        self.assertAlmostEqual(metrics["agent/candidate_reward_mean"], 0.5)

    def test_example2_config_data_and_dry_run(self) -> None:
        # Must reject config with unsupported process-credit keys
        with tempfile.TemporaryDirectory() as tmp:
            bad_cfg = Path(tmp) / "bad.yaml"
            bad_cfg.write_text("agent_config:\n  max_turns: 4\n  format_reward: 0.0\n  credit: {}\n")
            with self.assertRaises(ValueError):
                resolve(bad_cfg, {})

            # data.py accepts tasks WITHOUT evidence_targets
            src = Path(tmp) / "tasks.jsonl"
            dst = Path(tmp) / "out.jsonl"
            row = {
                "prompt": "Find the dog",
                "label": "A",
                "metadata": {
                    "task_id": "gspo-1",
                    "media_id": "m1",
                    "source_name": "unit",
                    "answer_type": "mcq",
                    "media": {"path": "m.parquet", "contract": "c.json", "row_group": 0},
                },
            }
            src.write_text(json.dumps(row) + "\n")
            self.assertEqual(prepare_jsonl(src, dst, check_files=False), 1)

        run_sh = EXAMPLE_DIR / "run.sh"
        subprocess.run(["bash", "-n", str(run_sh)], check=True)
        env = {**os.environ, "DRY_RUN": "1", "PYTHON_BIN": sys.executable}
        proc = subprocess.run(["bash", str(run_sh)], env=env, capture_output=True, text=True, check=True)
        cmd = proc.stdout.strip()
        self.assertIn("--advantage-estimator gspo", cmd)
        self.assertIn("--flush-cache-interval", cmd)
        self.assertIn("fully_async_rollout.generate_rollout_fully_async", cmd)
        self.assertIn("examples.qwen3_omni_agent_gspo.rollout.generate", cmd)
        self.assertNotIn("--use-turn-level-gspo", cmd)
        for bad in FORBIDDEN_DEPRECATED_TOKENS:
            self.assertNotIn(bad, cmd, f"deprecated token {bad!r} found in Example 2 run.sh")


    def test_shared_intervals_math_and_agent_cleanups(self) -> None:
        from slime_ext.agent import intervals
        from slime_ext.agent import protocol as agent_protocol
        from slime_ext.agent import rollout as agent_rollout

        self.assertEqual(intervals.merge_intervals([(3.0, 5.0), (1.0, 3.0), (2.0, 2.5)]), [(1.0, 5.0)])
        self.assertAlmostEqual(intervals.total_duration([(0.0, 2.0), (1.0, 4.0)]), 4.0)
        self.assertAlmostEqual(intervals.overlap_duration([(0.0, 3.0)], [(2.0, 5.0)]), 1.0)
        self.assertAlmostEqual(intervals.temporal_iou([(0.0, 3.0)], [(2.0, 5.0)]), 1.0 / 5.0)
        self.assertAlmostEqual(intervals.w1_distance([(0.0, 2.0)], [(2.0, 4.0)]), 2.0)
        self.assertAlmostEqual(intervals.w1_distance([(0.0, 2.0)], [(0.0, 2.0)]), 0.0)

        # Dead types removed and global default executor never mutated
        self.assertFalse(hasattr(agent_protocol, "TurnBoundary"))
        self.assertFalse(hasattr(agent_rollout, "MediaExecutionError"))
        rollout_src = (ROOT / "slime_ext/agent/rollout.py").read_text(encoding="utf-8")
        self.assertNotIn("set_default_executor", rollout_src)

if __name__ == "__main__":
    unittest.main()
