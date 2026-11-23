from __future__ import annotations

import logging
from copy import deepcopy

import torch
from megatron.core.models.common.embeddings.rope_utils import (
    get_pos_emb_on_this_cp_rank,
)
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    MultimodalRotaryEmbedding,
)
from megatron.core.models.gpt import GPTModel
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import make_viewless_tensor

logger = logging.getLogger(__name__)


def compact_response_hidden_states(
    hidden_states: torch.Tensor,
    response_rows: torch.Tensor,
) -> torch.Tensor:
    """Select packed next-token response rows from a full ``[T, 1, H]`` stream."""
    if hidden_states.ndim != 3 or hidden_states.size(1) != 1:
        raise ValueError(f"Expected hidden states shaped [T, 1, H], got {tuple(hidden_states.shape)}")
    if not torch.is_tensor(response_rows):
        raise TypeError("Response rows must be a tensor mask")
    if response_rows.ndim != 2 or response_rows.size(0) != 1:
        raise ValueError(
            f"Expected response row mask shaped [1, T], got {tuple(response_rows.shape)}"
        )
    if hidden_states.size(0) != response_rows.size(1):
        raise ValueError(
            "Response row mask and gathered hidden stream disagree: "
            f"{response_rows.size(1)} != {hidden_states.size(0)}"
        )
    row_indices = torch.nonzero(response_rows.reshape(-1), as_tuple=False).reshape(-1)
    return hidden_states.index_select(0, row_indices).contiguous()


class Qwen3OmniMultimodalRotaryEmbedding(MultimodalRotaryEmbedding):
    """Qwen3's interleaved temporal/height/width MRoPE layout."""

    def forward(
        self,
        position_ids: torch.Tensor,
        mrope_section: list[int],
        packed_seq: bool = False,
        cp_group=None,
        return_raw_freqs: bool = False,
    ) -> torch.Tensor:
        if return_raw_freqs:
            raise NotImplementedError(
                "Qwen3OmniMultimodalRotaryEmbedding does not support fused RoPE "
                "(return_raw_freqs=True); set config.apply_rope_fusion = False."
            )
        if self.rotary_interleaved:
            raise ValueError("Qwen3 MRoPE requires non-interleaved rotary pairs")
        if position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, -1, -1)
        seq = position_ids.to(device=self.inv_freq.device, dtype=torch.float32)
        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        inv_freq = self.inv_freq[None, None, :, None].expand(3, seq.shape[1], -1, 1)
        freqs = (inv_freq @ seq[:, :, None, :]).transpose(2, 3)

        mixed_freqs = freqs[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            section = slice(offset, mrope_section[axis] * 3, 3)
            mixed_freqs[..., section] = freqs[axis, ..., section]

        emb = torch.cat((mixed_freqs, mixed_freqs), dim=-1)[..., None, :]
        emb = emb.transpose(0, 1).contiguous()
        cp_group = cp_group or self.cp_group
        if cp_group is not None and cp_group.size() > 1 and not packed_seq:
            emb = get_pos_emb_on_this_cp_rank(emb, 0, cp_group)
        return emb


class Qwen3OmniTransformerLayer(TransformerLayer):
    """A standard MCore decoder layer with an optional DeepStack residual."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        output, _ = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            context=None,
            context_mask=None,
            **kwargs,
        )
        deepstack_index = self.layer_number - 1
        if context is None or deepstack_index >= context.shape[0]:
            return output, context
        if context_mask is None:
            raise ValueError("DeepStack embeddings require a visual token mask")

        output_bsh = output.transpose(0, 1).contiguous()
        visual_embeds = context[deepstack_index]
        if output_bsh[context_mask].shape != visual_embeds.shape:
            raise ValueError("DeepStack embeddings do not match the visual tokens")
        output_bsh[context_mask] = output_bsh[context_mask].clone() + visual_embeds
        output = output_bsh.transpose(0, 1).contiguous()
        output = make_viewless_tensor(
            inp=output,
            requires_grad=output.requires_grad,
            keep_graph=True,
        )
        return output, context


def _with_deepstack_layers(
    spec: TransformerBlockSubmodules,
) -> TransformerBlockSubmodules:
    if not isinstance(spec, TransformerBlockSubmodules):
        raise TypeError("Qwen3-Omni requires a native MCore decoder block spec")

    spec = deepcopy(spec)
    for layer_spec in spec.layer_specs:
        layer_spec.module = Qwen3OmniTransformerLayer
    return spec


class Qwen3OmniGPTModel(GPTModel):
    """Native MCore GPT with Qwen3 MRoPE and DeepStack injection."""

    def __init__(
        self,
        *args,
        transformer_layer_spec: TransformerBlockSubmodules,
        rotary_percent: float = 1.0,
        rotary_base: int = 10_000,
        seq_len_interpolation_factor: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            transformer_layer_spec=_with_deepstack_layers(transformer_layer_spec),
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            **kwargs,
        )
        self.rotary_pos_emb = Qwen3OmniMultimodalRotaryEmbedding(
            kv_channels=self.config.kv_channels,
            rotary_percent=rotary_percent,
            rotary_interleaved=self.config.rotary_interleaved,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            rotary_base=rotary_base,
            cp_group=self.pg_collection.cp,
        )

    def forward(
        self,
        *args,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        extra_block_kwargs: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if deepstack_visual_embeds:
            if visual_pos_masks is None:
                raise ValueError("DeepStack embeddings require a visual token mask")
            extra_block_kwargs = dict(extra_block_kwargs or {})
            if "context" in extra_block_kwargs or "context_mask" in extra_block_kwargs:
                raise ValueError("DeepStack reserves decoder context inputs")
            extra_block_kwargs["context"] = torch.stack(deepstack_visual_embeds)
            extra_block_kwargs["context_mask"] = visual_pos_masks

        return super().forward(
            *args,
            extra_block_kwargs=extra_block_kwargs,
            **kwargs,
        )

    def _postprocess(
        self,
        hidden_states,
        *,
        labels=None,
        loss_mask=None,
        mtp_in_postprocess=None,
        mtp_kwargs=None,
        inference_context=None,
        **kwargs,
    ):
        if not getattr(self.config, "response_only_lm_head", False) or not self.post_process:
            return super()._postprocess(
                hidden_states=hidden_states,
                labels=labels,
                loss_mask=loss_mask,
                mtp_in_postprocess=mtp_in_postprocess,
                mtp_kwargs=mtp_kwargs,
                inference_context=inference_context,
                **kwargs,
            )
        if labels is not None:
            raise ValueError("response-only LM head expects Slime's labels=None policy path")
        if loss_mask is None:
            raise ValueError("response-only LM head requires the packed response row mask")
        if mtp_in_postprocess or (mtp_kwargs and mtp_kwargs.get("mtp_labels") is not None):
            raise ValueError("response-only LM head is not compatible with MTP")
        if inference_context is not None:
            raise ValueError("response-only LM head is a training/log-prob path, not Megatron inference")
        sequence_parallel_override = False
        if self.output_layer.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                # The non-SP output head already sums its TP input gradients.
                tensor_parallel_output_grad=False,
                group=self.pg_collection.tp,
            )
            self.output_layer.sequence_parallel = False
            sequence_parallel_override = True

        try:
            packed_rows = hidden_states.size(0)
            response_hidden_states = compact_response_hidden_states(hidden_states, loss_mask)
            empty_response = response_hidden_states.size(0) == 0
            # Keep the head and attention in autograd for zero-response batches
            # under both cp_size == 1 and cp_size > 1 without sending a zero-row
            # GEMM to TE.
            hidden_states = hidden_states[:1] if empty_response else response_hidden_states
            if not getattr(self, "_response_only_lm_head_logged", False):
                logger.info(
                    "response-only LM head active: projected_rows=%d packed_rows=%d",
                    response_hidden_states.size(0),
                    packed_rows,
                )
                self._response_only_lm_head_logged = True
            output = super()._postprocess(
                hidden_states=hidden_states,
                labels=labels,
                loss_mask=loss_mask,
                mtp_in_postprocess=mtp_in_postprocess,
                mtp_kwargs=mtp_kwargs,
                inference_context=inference_context,
                **kwargs,
            )
            return output[:, :0].contiguous() if empty_response else output
        finally:
            if sequence_parallel_override:
                self.output_layer.sequence_parallel = True
