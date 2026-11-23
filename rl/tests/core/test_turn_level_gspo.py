"""CPU numerical, gradient, and Context Parallelism (CP) alignment tests for Turn-Level GSPO."""

from __future__ import annotations

import functools
import os
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SLIME_ROOT = Path(os.environ.get("SLIME_DIR", str(REPO_ROOT / "third_party" / "slime")))

for _p in (str(SLIME_ROOT), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class TurnLevelGSPOTests(unittest.TestCase):
    def test_drifting_tool_turn_clipped_while_in_trust_region_answer_turn_unclipped(self):
        """Prove TG-GSPO isolates per-turn clipping so a drifting tool turn does NOT clip an answer turn."""
        from slime_ext.backends.megatron.loss import compute_turn_level_gspo_loss

        old_lp = torch.zeros(10, dtype=torch.float32)
        log_p = torch.tensor(
            [0.8, 0.8, 0.8, 0.8, 5.0, -5.0, 0.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
            requires_grad=True,
        )
        loss_mask = torch.tensor([1, 1, 1, 1, 0, 0, 1, 1, 1, 1], dtype=torch.int)
        turn_ids = torch.tensor([0, 0, 0, 0, -1, -1, 1, 1, 1, 1], dtype=torch.long)
        advantages = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)

        reduced_loss, _, clipfrac, ppo_kl = compute_turn_level_gspo_loss(
            full_log_probs=[log_p],
            full_old_log_probs=[old_lp],
            local_log_probs=[log_p],
            advantages=[advantages],
            loss_masks=[loss_mask],
            turn_ids=[turn_ids],
            eps_clip=0.2,
            eps_clip_high=0.28,
        )

        self.assertTrue(torch.allclose(ppo_kl[:4], torch.full((4,), -0.8), atol=1e-6))
        self.assertTrue(torch.allclose(ppo_kl[4:6], torch.zeros(2), atol=1e-6))
        self.assertTrue(torch.allclose(ppo_kl[6:], torch.zeros(4), atol=1e-6))
        self.assertEqual(clipfrac[:4].tolist(), [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(clipfrac[6:].tolist(), [0.0, 0.0, 0.0, 0.0])

        reduced_loss.backward()
        self.assertTrue(torch.allclose(log_p.grad[:4], torch.zeros(4), atol=1e-7))
        self.assertTrue(torch.allclose(log_p.grad[4:6], torch.zeros(2), atol=1e-7))
        self.assertTrue(torch.allclose(log_p.grad[6:], torch.full((4,), -1.0 / 8.0), atol=1e-6))

    def test_turn_weights_and_analytic_gradients(self):
        from slime_ext.backends.megatron.loss import compute_turn_level_gspo_loss

        old_lp = torch.zeros(8, dtype=torch.float32)
        log_p = torch.tensor([0.05, -0.05, 0.01, -0.01, 0.02, -0.02, 0.0, 0.0], dtype=torch.float32, requires_grad=True)
        loss_mask = torch.ones(8, dtype=torch.int)
        turn_ids = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)
        adv = torch.tensor([2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        turn_weights = torch.tensor(
            [0.25, 0.25, 0.5 / 6, 0.5 / 6, 0.5 / 6, 0.5 / 6, 0.5 / 6, 0.5 / 6],
            dtype=torch.float32,
        )

        loss, _, clipfrac, _ = compute_turn_level_gspo_loss(
            full_log_probs=[log_p],
            full_old_log_probs=[old_lp],
            local_log_probs=[log_p],
            advantages=[adv],
            loss_masks=[loss_mask],
            turn_ids=[turn_ids],
            turn_weights=[turn_weights],
            eps_clip=0.2,
            eps_clip_high=0.2,
        )
        self.assertAlmostEqual(loss.item(), -1.5, places=6)
        self.assertTrue(torch.all(clipfrac == 0))
        loss.backward()
        self.assertTrue(torch.allclose(log_p.grad[:2], torch.full((2,), -0.5), atol=1e-6))
        self.assertTrue(torch.allclose(log_p.grad[2:], torch.full((6,), -1.0 / 12.0), atol=1e-6))

    def test_cp_zigzag_slicing_exact_equivalence_loss_and_gradients(self):
        """Verify CP=2 and CP=4 TG-GSPO loss & gradients match CP=1 identically."""
        from slime_ext.backends.megatron.loss import (
            compute_turn_level_gspo_loss,
            slice_response_tensor_with_cp,
        )

        torch.manual_seed(1234)
        total_lengths = [25, 31]
        response_lengths = [14, 19]
        old_lps = [torch.randn(r) * 0.2 for r in response_lengths]
        base_lps = [o + torch.randn_like(o) * 0.15 for o in old_lps]
        base_lps[0][:4] = old_lps[0][:4] + 0.9
        loss_masks = [
            torch.tensor([1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1], dtype=torch.int),
            torch.tensor([1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int),
        ]
        turn_ids = [
            torch.tensor([0, 0, 0, 0, -1, -1, 1, 1, 1, 1, -1, 2, 2, 2], dtype=torch.long),
            torch.tensor([0, 0, 0, -1, 1, 1, 1, 1, 1, -1, -1, 2, 2, 2, 2, 2, 2, 2, 2], dtype=torch.long),
        ]
        advs = [torch.randn(r) for r in response_lengths]
        turn_weights = [torch.rand(r) + 0.1 for r in response_lengths]

        ref_lps = [t.clone().detach().requires_grad_(True) for t in base_lps]
        ref_loss, _, _, _ = compute_turn_level_gspo_loss(
            full_log_probs=ref_lps,
            full_old_log_probs=old_lps,
            local_log_probs=ref_lps,
            advantages=advs,
            loss_masks=loss_masks,
            turn_ids=turn_ids,
            turn_weights=turn_weights,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            eps_clip=0.2,
            eps_clip_high=0.28,
        )
        ref_loss.backward()
        ref_grads = [t.grad.clone() for t in ref_lps]

        for cp_size in (2, 4):
            cp_lps = [t.clone().detach().requires_grad_(True) for t in base_lps]
            rank_losses = []
            for cp_rank in range(cp_size):
                slicer = functools.partial(slice_response_tensor_with_cp, cp_size=cp_size, cp_rank=cp_rank)
                local_lps = [
                    slicer(lp, t_len, r_len)
                    for lp, t_len, r_len in zip(cp_lps, total_lengths, response_lengths, strict=True)
                ]
                local_advs = [
                    slicer(a, t_len, r_len)
                    for a, t_len, r_len in zip(advs, total_lengths, response_lengths, strict=True)
                ]
                rank_loss, _, _, _ = compute_turn_level_gspo_loss(
                    full_log_probs=cp_lps,
                    full_old_log_probs=old_lps,
                    local_log_probs=local_lps,
                    advantages=local_advs,
                    loss_masks=loss_masks,
                    turn_ids=turn_ids,
                    turn_weights=turn_weights,
                    total_lengths=total_lengths,
                    response_lengths=response_lengths,
                    eps_clip=0.2,
                    eps_clip_high=0.28,
                    slice_fn=slicer,
                )
                rank_losses.append(rank_loss)

            total_cp_loss = sum(rank_losses)
            total_cp_loss.backward()
            self.assertAlmostEqual(total_cp_loss.item(), ref_loss.item(), places=5)
            for sample_idx in range(len(base_lps)):
                self.assertTrue(
                    torch.allclose(cp_lps[sample_idx].grad, ref_grads[sample_idx], atol=1e-5, rtol=1e-5),
                    f"CP={cp_size} grad mismatch on sample {sample_idx}",
                )

    def test_single_turn_matches_sequence_gspo_and_all_masked_sample_safe(self):
        from slime_ext.backends.megatron.loss import compute_turn_level_gspo_loss

        dead_lp = torch.randn(5, requires_grad=True)
        dead_loss, _, _, _ = compute_turn_level_gspo_loss(
            full_log_probs=[dead_lp],
            full_old_log_probs=[torch.zeros(5)],
            local_log_probs=[dead_lp],
            advantages=[torch.ones(5)],
            loss_masks=[torch.zeros(5, dtype=torch.int)],
            turn_ids=[torch.full((5,), -1, dtype=torch.long)],
            eps_clip=0.2,
            eps_clip_high=0.2,
        )
        self.assertEqual(dead_loss.item(), 0.0)
        dead_loss.backward()
        self.assertTrue(torch.isfinite(dead_lp.grad).all())
        self.assertTrue(torch.all(dead_lp.grad == 0))


if __name__ == "__main__":
    unittest.main()
