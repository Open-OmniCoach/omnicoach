"""CPU contract tests for Example 1: qwen3_omni_rl (pure single-turn multimodal RL)."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

from examples.qwen3_omni_rl.configure import resolve
from examples.qwen3_omni_rl.data import build_multimodal_messages, prepare_jsonl
from examples.qwen3_omni_rl.reward import (
    score,
    score_av_match_response,
    score_mcq_response,
    score_temporal_iou_response,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = ROOT / "examples" / "qwen3_omni_rl"
UPSTREAM_SLIME_ARGS = ROOT / "third_party" / "slime" / "slime" / "utils" / "arguments.py"

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


class TestQwen3OmniRLContract(unittest.TestCase):
    def test_configure_defaults_and_overrides(self) -> None:
        cfg = resolve(EXAMPLE_DIR / "config.yaml", {})
        self.assertEqual(cfg["qwen3_omni_rl"]["advantage_estimator"], "grpo")
        self.assertEqual(cfg["qwen3_omni_rl"]["ablation_mode"], "none")

        over = resolve(
            EXAMPLE_DIR / "config.yaml",
            {"ADVANTAGE_ESTIMATOR": "gspo", "REWARD_TYPE": "temporal_iou", "ABLATION_MODE": "no-audio"},
        )
        self.assertEqual(over["qwen3_omni_rl"]["advantage_estimator"], "gspo")
        self.assertEqual(over["qwen3_omni_rl"]["reward_type"], "temporal_iou")
        self.assertEqual(over["qwen3_omni_rl"]["ablation_mode"], "no-audio")

        with self.assertRaises(ValueError):
            resolve(EXAMPLE_DIR / "config.yaml", {"ADVANTAGE_ESTIMATOR": "invalid_algo"})
        with self.assertRaises(ValueError):
            resolve(EXAMPLE_DIR / "config.yaml", {"ABLATION_MODE": "bad-mode"})

    def test_run_sh_honors_custom_config_and_ablation_arg(self) -> None:
        run_sh = EXAMPLE_DIR / "run.sh"
        with tempfile.TemporaryDirectory() as tmp:
            custom_cfg = Path(tmp) / "custom.yaml"
            custom_cfg.write_text(
                "qwen3_omni_rl:\n"
                "  advantage_estimator: gspo\n"
                "  reward_type: mcq\n"
                "  ablation_mode: no-audio\n",
                encoding="utf-8",
            )
            env = {**os.environ, "DRY_RUN": "1", "PYTHON_BIN": sys.executable, "RL_CONFIG": str(custom_cfg)}
            env.pop("ADVANTAGE_ESTIMATOR", None)
            proc = subprocess.run(["bash", str(run_sh)], env=env, capture_output=True, text=True, check=True)
            cmd = proc.stdout.strip()
            self.assertIn("--advantage-estimator gspo", cmd)
        self.assertIn('DATA_PREPARE_ARGS=(--ablation-mode "${RESOLVED_ABLATION}")', run_sh.read_text(encoding="utf-8"))
        self.assertIn("DATA_PREPARE_ARGS", (ROOT / "examples" / "_common" / "launch_qwen3_omni.sh").read_text(encoding="utf-8"))

    def test_data_messages_ablation_and_jsonl(self) -> None:
        row = {
            "prompt": "<image><audio>What is happening?",
            "images": ["img.png"],
            "audios": ["aud.wav"],
            "label": "B",
        }
        msgs = build_multimodal_messages(row, ablation_mode="none")
        self.assertEqual([p["type"] for p in msgs[0]["content"]], ["image", "audio", "text"])

        no_audio = build_multimodal_messages(row, ablation_mode="no-audio")
        self.assertEqual([p["type"] for p in no_audio[0]["content"]], ["image", "text"])

        text_only = build_multimodal_messages(row, ablation_mode="text-only")
        self.assertEqual([p["type"] for p in text_only[0]["content"]], ["text"])

        # Mismatched leftover media raises ValueError
        with self.assertRaises(ValueError):
            build_multimodal_messages(
                {"prompt": "No placeholder", "images": ["extra.png"], "label": "A"}
            )

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.jsonl"
            dst = Path(tmp) / "out.jsonl"
            src.write_text(json.dumps(row) + "\n")
            count = prepare_jsonl(src, dst)
            self.assertEqual(count, 1)
            out_row = json.loads(dst.read_text().strip())
            self.assertEqual(out_row["label"], "B")
            self.assertEqual(out_row["metadata"]["task_id"], "row-1")

    def test_reward_mcq_av_and_temporal_iou(self) -> None:
        self.assertEqual(score_mcq_response("<answer>B</answer>", "b"), 1.0)
        self.assertEqual(score_mcq_response("<answer>C</answer>", "B"), 0.0)
        self.assertEqual(score_av_match_response("<answer>mismatch</answer>", "mismatch"), 1.0)
        self.assertEqual(score_av_match_response("<answer>match</answer>", "mismatch"), 0.0)
        self.assertAlmostEqual(
            score_temporal_iou_response("<answer>[10s, 20s]</answer>", [15.0, 25.0]),
            5.0 / 15.0,
            places=6,
        )

        args = types.SimpleNamespace(qwen3_omni_rl={"reward_type": "auto"})
        s1 = types.SimpleNamespace(response="<answer>A</answer>", label="A", metadata={"answer_type": "mcq"})
        s2 = types.SimpleNamespace(response="match", label="match", metadata={})
        s3 = types.SimpleNamespace(response="from 2s to 6s", label=[2.0, 6.0], metadata={})
        batch_scores = asyncio.run(score(args, [s1, s2, s3]))
        self.assertEqual(batch_scores, [1.0, 1.0, 1.0])
        single = asyncio.run(score(args, s1))
        self.assertEqual(single, 1.0)

    def test_run_sh_syntax_dry_run_and_upstream_flags(self) -> None:
        run_sh = EXAMPLE_DIR / "run.sh"
        subprocess.run(["bash", "-n", str(run_sh)], check=True)

        env = {**os.environ, "DRY_RUN": "1", "PYTHON_BIN": sys.executable}
        proc = subprocess.run(
            ["bash", str(run_sh)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        cmd = proc.stdout.strip()
        self.assertIn("--advantage-estimator grpo", cmd)
        self.assertIn("--flush-cache-interval", cmd)
        self.assertIn("examples.qwen3_omni_rl.reward.score", cmd)
        self.assertNotIn("--use-turn-level-gspo", cmd)
        self.assertIn("-m slime_ext.train", cmd)
        self.assertIn("--ray-address", cmd)
        self.assertIn("--ray-env-vars-json", cmd)
        self.assertNotIn("DRIVER_CODE", cmd)
        self.assertNotIn(" -c ", cmd)
        for bad in FORBIDDEN_DEPRECATED_TOKENS:
            self.assertNotIn(bad, cmd, f"deprecated token {bad!r} found in Example 1 run.sh")

        if UPSTREAM_SLIME_ARGS.is_file():
            upstream_flags = set(re.findall(r"--[a-zA-Z0-9_-]+", UPSTREAM_SLIME_ARGS.read_text()))
            tokens = shlex.split(cmd)
            emitted_slime_flags = [
                t for t in tokens
                if t.startswith("--") and t in upstream_flags
            ]
            # Confirm core latest-slime flags actually exist in upstream arguments.py
            for must_exist in (
                "--flush-cache-interval",
                "--rollout-function-path",
                "--custom-rm-path",
                "--dynamic-sampling-filter-path",
                "--rollout-data-transport",
                "--advantage-estimator",
            ):
                self.assertIn(must_exist, upstream_flags)
                self.assertIn(must_exist, emitted_slime_flags)


if __name__ == "__main__":
    unittest.main()
