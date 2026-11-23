"""Keep supplied Omni token IDs exact while reusing SGLang's media processing."""

import logging
from array import array
from collections import defaultdict

import torch
from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    _get_feat_extract_output_lengths,
)

logger = logging.getLogger(__name__)


class ExactTokenQwen3OmniProcessor(QwenVLImageProcessor):
    """Avoid decode/re-encode drift in teacher forcing and partial resumes.

    The pinned QwenVL processor decodes token IDs to run the HF media processor.
    Generated BPE tokens need not be its canonical re-encoding. Keep the media
    features, but rebuild the token layout with SGLang's native token expander.
    """

    models = [Qwen3OmniMoeForConditionalGeneration]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Qwen3-Omni exact-token media processor enabled")

    def compose_image_artifacts(self, input_text, artifacts):
        """Disable SGLang 0.5.21's image-artifact cache fast path.

        Returning ``None`` forces ``QwenVLImageProcessor.process_mm_data_async``
        to fall back to ``_process_mm_data_uncached`` so exact pretokenized IDs
        are never bypassed by cached prompt templates.
        """
        return None

    async def process_mm_data_async(
        self, image_data, input_text, request_obj, *args, **kwargs
    ):
        if hasattr(self, "_process_mm_data_uncached") and isinstance(
            input_text, (list, array)
        ):
            return await self._process_mm_data_uncached(
                image_data, input_text, request_obj, *args, **kwargs
            )
        return await super().process_mm_data_async(
            image_data, input_text, request_obj, *args, **kwargs
        )

    async def legacy_load_mm_data(self, prompt, *args, **kwargs):
        output = await super().legacy_load_mm_data(prompt, *args, **kwargs)
        if isinstance(prompt, array):
            output.input_ids = prompt.tolist()
        elif isinstance(prompt, list):
            output.input_ids = prompt
        return output

    def process_and_combine_mm_data(self, base_output, mm_tokens, **kwargs):
        items, input_ids, features = super().process_and_combine_mm_data(
            base_output, mm_tokens, **kwargs,
        )
        if base_output.input_ids is None:
            return items, input_ids, features

        audio_mask = features.get("feature_attention_mask")
        audio_lengths = (
            _get_feat_extract_output_lengths(audio_mask.sum(dim=1))
            if audio_mask is not None else None
        )
        exact_ids, offsets, modalities = self.build_input_ids(
            base_output.input_ids,
            features.get("image_grid_thw"),
            features.get("video_grid_thw"),
            audio_lengths,
        )
        per_modality = defaultdict(list)
        for offset, modality in zip(offsets, modalities, strict=True):
            per_modality[modality].append(offset)
        cursors = {modality: iter(rows) for modality, rows in per_modality.items()}
        for item in items:
            item.offsets = [next(cursors[item.modality]) for _ in item.offsets]
        return items, torch.tensor(exact_ids, dtype=input_ids.dtype), features
