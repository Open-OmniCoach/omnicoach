# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable

import torch
from megatron.core import InferenceParams, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerConfig as Qwen3OmniMoeThinkerConfigHF,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder as Qwen3OmniMoeAudioEncoderHF,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoder as Qwen3OmniMoeVisionEncoderHF,
)

from slime_ext.backends.megatron.packed import (
    count_audio_token_runs as _count_audio_token_runs,
)
from slime_ext.backends.megatron.packed import (
    get_real_cu_seqlens_q,
    pack_padded_batch_position_ids,
    packed_attention_mask,
    packed_thd_to_padded_batch,
)
from slime_ext.models.qwen3_omni.parallel import (
    encode_frozen_vision_cp,
    mark_replicated_parameters,
    pack_cp_position_ids,
    select_packed_cp_deepstack,
    shard_deepstack_for_model_parallel,
    split_data_for_context_parallel,
    unpack_cp_full_tokens,
)
from slime_ext.models.qwen3_omni.parallel import (
    normalize_visual_outputs as _normalize_visual_outputs,
)
from slime_ext.models.qwen3_omni.rope import (
    get_qwen3_omni_audio_output_lengths,
    get_rope_index,
)
from slime_ext.models.qwen3_omni.text import Qwen3OmniGPTModel


def _deep_getattr(module: torch.nn.Module, attr_path: str) -> torch.nn.Module:
    value = module
    for attr in attr_path.split("."):
        value = getattr(value, attr)
    return value


def _patch_get_input_embeddings(module: torch.nn.Module, attr_path: str) -> None:
    """Match ms-swift's tower patching for gradient-checkpoint input hooks."""

    def _get_input_embeddings(self) -> torch.nn.Module:
        return _deep_getattr(self, attr_path)

    module.get_input_embeddings = _get_input_embeddings.__get__(module, module.__class__)


def _build_text_only_mrope_position_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create text-only multimodal rope ids shaped [3, batch, seq]."""
    batch_size, seq_len = input_ids.shape
    if attention_mask is None:
        base = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        base = base.unsqueeze(0).expand(batch_size, -1)
    else:
        attention_mask = attention_mask.to(input_ids.device).bool()
        base = attention_mask.long().cumsum(-1) - 1
        base.masked_fill_(~attention_mask, 1)
    return torch.stack([base, base, base], dim=0)


def _configure_multimodal_attn_impl(config: object, attn_impl: str | None) -> None:
    """Apply a requested attention implementation to HF multimodal configs."""
    if not attn_impl or attn_impl == "auto":
        return

    config._attn_implementation = attn_impl
    config._attn_implementation_internal = attn_impl
    config.attn_implementation = attn_impl

    use_flash_attn = attn_impl in {"flash_attn", "flash_attention_2"}
    config.use_flash_attn = use_flash_attn
    config._use_flash_attention_2 = use_flash_attn
    config._flash_attn_2_enabled = use_flash_attn


def _enable_multimodal_gradient_checkpointing(module: torch.nn.Module) -> None:
    """Best-effort enable gradient checkpointing for HF multimodal towers."""
    if hasattr(module, "gradient_checkpointing_enable"):
        try:
            module.gradient_checkpointing_enable()
        except TypeError:
            module.gradient_checkpointing_enable({})
        except NotImplementedError:
            pass

    if hasattr(module, "enable_input_require_grads"):
        try:
            module.enable_input_require_grads()
        except AttributeError:
            pass
        except NotImplementedError:
            pass


def _trim_feature_sequence(
    features: torch.Tensor | None,
    multiscale_features: list[torch.Tensor] | None,
    expected_tokens: int,
    feature_name: str,
) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
    if features is None:
        return None, multiscale_features

    produced_tokens = int(features.shape[0])
    if expected_tokens > produced_tokens:
        raise ValueError(
            f"{feature_name} placeholders exceed produced features: "
            f"expected={expected_tokens}, produced={produced_tokens}"
        )
    if expected_tokens == produced_tokens:
        return features, multiscale_features

    trimmed_features = features[:expected_tokens]
    trimmed_multiscale = None
    if multiscale_features is not None:
        trimmed_multiscale = [feature[:expected_tokens] for feature in multiscale_features]
    return trimmed_features, trimmed_multiscale


def _trim_visual_features_per_sample(
    features: torch.Tensor | None,
    multiscale_features: list[torch.Tensor] | None,
    *,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    expected_per_sample: list[int],
    item_counts: list[int] | None,
    feature_name: str,
) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
    """Trim truncated visual features per sample so sample i never steals sample i+1 features."""
    if features is None:
        return None, multiscale_features

    total_expected = sum(expected_per_sample)
    total_produced = int(features.shape[0])
    if len(expected_per_sample) <= 1:
        return _trim_feature_sequence(features, multiscale_features, total_expected, feature_name)

    if item_counts is None:
        if total_expected == total_produced:
            return features, multiscale_features
        raise ValueError(
            f"{feature_name} count mismatch in multi-sample batch without per-sample item_counts "
            f"(expected={total_expected}, produced={total_produced}); refusing cross-sample tail trim."
        )

    if len(item_counts) != len(expected_per_sample):
        raise ValueError(
            f"{feature_name} item_counts length {len(item_counts)} != batch size {len(expected_per_sample)}"
        )

    merge_area = spatial_merge_size * spatial_merge_size
    tokens_per_grid = (grid_thw.prod(dim=1) // merge_area).tolist()
    if sum(item_counts) != len(tokens_per_grid):
        raise ValueError(
            f"{feature_name} sum(item_counts)={sum(item_counts)} != grid_thw rows {len(tokens_per_grid)}"
        )

    feat_chunks: list[torch.Tensor] = []
    ms_chunks: list[list[torch.Tensor]] | None = (
        [[] for _ in multiscale_features] if multiscale_features is not None else None
    )
    grid_cursor = 0
    feat_cursor = 0
    for sample_idx, (expected_i, num_items_i) in enumerate(zip(expected_per_sample, item_counts, strict=True)):
        produced_i = sum(tokens_per_grid[grid_cursor : grid_cursor + num_items_i])
        grid_cursor += num_items_i
        if expected_i > produced_i:
            raise ValueError(
                f"{feature_name} placeholders exceed produced features in sample {sample_idx}: "
                f"expected={expected_i}, produced={produced_i}"
            )
        if expected_i > 0:
            feat_chunks.append(features[feat_cursor : feat_cursor + expected_i])
            if ms_chunks is not None and multiscale_features is not None:
                for lvl, ms in enumerate(multiscale_features):
                    ms_chunks[lvl].append(ms[feat_cursor : feat_cursor + expected_i])
        feat_cursor += produced_i

    if feat_cursor != total_produced:
        raise ValueError(
            f"{feature_name} grid_thw token total {feat_cursor} != produced features {total_produced}"
        )

    trimmed_features = (
        torch.cat(feat_chunks, dim=0)
        if feat_chunks
        else features.new_zeros((0, *features.shape[1:]))
    )
    trimmed_multiscale = None
    if ms_chunks is not None and multiscale_features is not None:
        trimmed_multiscale = [
            torch.cat(chunks, dim=0) if chunks else ms.new_zeros((0, *ms.shape[1:]))
            for chunks, ms in zip(ms_chunks, multiscale_features, strict=True)
        ]
    return trimmed_features, trimmed_multiscale


def _validate_deepstack_stage_layers(pre_process: bool, visual, language_model) -> None:
    """Fail fast if PP stage 0 holds fewer decoder layers than DeepStack levels."""
    if not pre_process or visual is None or language_model is None:
        return
    deepstack_indexes = getattr(visual, "deepstack_visual_indexes", None)
    if deepstack_indexes is None:
        deepstack_indexes = getattr(getattr(visual, "config", None), "deepstack_visual_indexes", None)
    num_deepstack = len(deepstack_indexes) if deepstack_indexes else 0
    if num_deepstack == 0:
        return
    decoder = getattr(language_model, "decoder", None)
    layers = getattr(decoder, "layers", None)
    if layers is not None and len(layers) < num_deepstack:
        raise ValueError(
            f"PP stage 0 has only {len(layers)} decoder layer(s), fewer than the "
            f"{num_deepstack} DeepStack visual residual levels (indexes={list(deepstack_indexes)}). "
            "Configure --decoder-first-pipeline-num-layers >= "
            f"{num_deepstack} so DeepStack residuals are not silently dropped."
        )


class Qwen3OmniThinkerModel(MegatronModule):
    """Qwen3-Omni thinker model supporting text, vision, and audio forward paths."""

    def __init__(
        self,
        language_transformer_config: TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        thinker_transformer_config: Qwen3OmniMoeThinkerConfigHF,
        parallel_output: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        add_decoder: bool = True,
        pg_collection: ProcessGroupCollection | None = None,
    ) -> None:
        super().__init__(config=language_transformer_config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_decoder = add_decoder

        self.visual = None
        self.audio_model = None
        self.language_model = None

        self.pg_collection = pg_collection
        self.image_token_id = language_transformer_config.image_token_id
        self.video_token_id = language_transformer_config.video_token_id
        self.audio_token_id = language_transformer_config.audio_token_id
        self.vision_start_token_id = language_transformer_config.vision_start_token_id
        self.audio_start_token_id = language_transformer_config.audio_start_token_id
        self.position_id_per_seconds = language_transformer_config.position_id_per_seconds
        self.thinker_transformer_config = thinker_transformer_config

        if self.pre_process:
            multimodal_attn_impl = getattr(language_transformer_config, "multimodal_attn_impl", "auto")
            _configure_multimodal_attn_impl(thinker_transformer_config.vision_config, multimodal_attn_impl)
            _configure_multimodal_attn_impl(thinker_transformer_config.audio_config, multimodal_attn_impl)

            vision_class = Qwen3OmniMoeVisionEncoderHF
            if getattr(language_transformer_config, "vision_batch_invariant_positions", False):
                from .vision import Qwen3OmniVisionEncoder

                vision_class = Qwen3OmniVisionEncoder
            self.visual = vision_class._from_config(thinker_transformer_config.vision_config)
            mark_replicated_parameters(self.visual)
            audio_class = Qwen3OmniMoeAudioEncoderHF
            if getattr(language_transformer_config, "audio_fixed_conv_batch", False):
                from .audio import Qwen3OmniAudioEncoder

                audio_class = Qwen3OmniAudioEncoder
            self.audio_model = audio_class._from_config(thinker_transformer_config.audio_config)
            mark_replicated_parameters(self.audio_model)
            _patch_get_input_embeddings(self.visual, "patch_embed")
            _patch_get_input_embeddings(self.audio_model, "conv_out")

            _configure_multimodal_attn_impl(
                getattr(self.visual, "config", self.thinker_transformer_config.vision_config), multimodal_attn_impl
            )
            _configure_multimodal_attn_impl(
                getattr(self.audio_model, "config", self.thinker_transformer_config.audio_config), multimodal_attn_impl
            )
            if getattr(language_transformer_config, "vit_gradient_checkpointing", False):
                _enable_multimodal_gradient_checkpointing(self.visual)
                _enable_multimodal_gradient_checkpointing(self.audio_model)

        self.language_model = Qwen3OmniGPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_transformer_config.vocab_size,
            max_sequence_length=language_transformer_config.language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type="mrope",
            rotary_percent=language_transformer_config.rotary_percent,
            pre_process=self.pre_process,
            post_process=self.post_process,
            rotary_base=language_transformer_config.rotary_base,
            fp16_lm_cross_entropy=language_transformer_config.fp16_lm_cross_entropy,
            share_embeddings_and_output_weights=language_transformer_config.share_embeddings_and_output_weights,
            scatter_embedding_sequence_parallel=False,
            pg_collection=pg_collection,
        )

        _validate_deepstack_stage_layers(self.pre_process, self.visual, self.language_model)
        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

    def shared_embedding_or_output_weight(self):
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for Qwen3Omni"
        if not self.pre_process:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        freeze_language_model: bool = False,
        freeze_vision_model: bool = False,
        freeze_audio_model: bool = False,
    ):
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.visual is not None:
            modules.append(self.visual)
        if freeze_audio_model and self.audio_model is not None:
            modules.append(self.audio_model)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    @staticmethod
    def _get_placeholder_mask(
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
        video_features: torch.FloatTensor | None = None,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_mask = input_ids == image_token_id
        video_mask = input_ids == video_token_id

        expanded_image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        expanded_video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds)

        if image_features is not None and inputs_embeds[expanded_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens={image_mask.sum()}, "
                f"features={image_features.shape[0]}"
            )
        if video_features is not None and inputs_embeds[expanded_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Video features and video tokens do not match: tokens={video_mask.sum()}, "
                f"features={video_features.shape[0]}"
            )

        return image_mask, video_mask

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        target_dtype = getattr(self.visual, "dtype", pixel_values.dtype)
        with torch.profiler.record_function("omni.visual_encoder"):
            if getattr(self.config, "vision_cp", False) and self.pg_collection.cp.size() > 1:
                return encode_frozen_vision_cp(
                    self.visual, pixel_values.to(dtype=target_dtype), image_grid_thw, self.pg_collection.cp
                )
            image_outputs = self.visual(pixel_values.to(dtype=target_dtype), grid_thw=image_grid_thw)
        return _normalize_visual_outputs(image_outputs)

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: torch.LongTensor | None = None,
        audio_feature_lengths: torch.LongTensor | None = None,
        expected_audio_token_counts: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)

        if audio_feature_lengths is None:
            raise ValueError("Either feature_attention_mask or audio_feature_lengths must be provided")

        target_dtype = getattr(self.audio_model, "dtype", input_features.dtype)
        with torch.profiler.record_function("omni.audio_encoder"):
            audio_outputs = self.audio_model(
                input_features.to(dtype=target_dtype),
                feature_lens=audio_feature_lengths,
            )
        audio_embeds = audio_outputs.last_hidden_state

        if expected_audio_token_counts is None:
            return audio_embeds

        produced_output_lengths = get_qwen3_omni_audio_output_lengths(audio_feature_lengths)
        expected_audio_token_counts = expected_audio_token_counts.to(produced_output_lengths.device)
        if expected_audio_token_counts.numel() != produced_output_lengths.numel():
            raise ValueError(
                "Audio placeholder spans and audio features do not match: "
                f"spans={expected_audio_token_counts.numel()}, features={produced_output_lengths.numel()}"
            )

        trimmed_embeds = []
        start = 0
        for produced_len, expected_len in zip(
            produced_output_lengths.tolist(),
            expected_audio_token_counts.tolist(),
            strict=True,
        ):
            produced_len = int(produced_len)
            expected_len = int(expected_len)
            end = start + produced_len
            if expected_len > produced_len:
                raise ValueError(
                    "Audio token placeholders exceed produced audio features: "
                    f"expected={expected_len}, produced={produced_len}"
                )
            trimmed_embeds.append(audio_embeds[start : start + expected_len])
            start = end

        if start != audio_embeds.shape[0]:
            raise ValueError(
                "Audio encoder output length does not match Qwen3-Omni feature lengths: "
                f"expected={start}, actual={audio_embeds.shape[0]}"
            )

        return (
            torch.cat(trimmed_embeds, dim=0)
            if trimmed_embeds
            else audio_embeds.new_zeros((0, audio_embeds.shape[-1]))
        )

    def _encode_and_trim_visual(
        self,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        *,
        encoder_fn: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, list[torch.Tensor]]],
        token_id: int,
        feature_name: str,
        grid_name: str,
        input_name: str,
        item_counts: list[int] | None,
        sample_tokens: torch.Tensor,
        sample_mask: torch.Tensor | None,
        target_device: torch.device,
        target_dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
        """Encode image or video inputs and trim per sample to match placeholder counts."""
        if pixel_values is None:
            return None, None
        if grid_thw is None:
            raise ValueError(f"{grid_name} is required when {input_name} is provided")
        embeds, multiscale = encoder_fn(pixel_values, grid_thw)
        embeds = embeds.to(target_device, target_dtype)
        multiscale = [embed.to(target_device, target_dtype) for embed in multiscale]
        token_matches = sample_tokens == token_id
        if sample_mask is not None:
            token_matches = token_matches & sample_mask
        expected_per_sample = [int(c) for c in token_matches.sum(dim=1).tolist()]
        return _trim_visual_features_per_sample(
            embeds,
            multiscale,
            grid_thw=grid_thw,
            spatial_merge_size=self.config.spatial_merge_size,
            expected_per_sample=expected_per_sample,
            item_counts=item_counts,
            feature_name=feature_name,
        )

    def _merge_multimodal_features(
        self,
        combined_embeddings: torch.Tensor,
        *,
        input_ids: torch.Tensor,
        is_packed: bool,
        packed_input_ids_padded: torch.Tensor | None,
        packed_rope_attention_mask: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
        input_features: torch.Tensor | None,
        feature_attention_mask: torch.Tensor | None,
        audio_feature_lengths: torch.Tensor | None,
        multimodal_num_items: dict[str, list[int]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        """Encode and scatter image, video, and audio features in a single S/B transpose."""
        if pixel_values is None and pixel_values_videos is None and input_features is None:
            return combined_embeddings, None, None

        inputs_embeds_bsh = combined_embeddings.transpose(0, 1).contiguous()
        visual_pos_masks: torch.Tensor | None = None
        deepstack_visual_embeds: list[torch.Tensor] | None = None

        if pixel_values is not None or pixel_values_videos is not None:
            sample_tokens = packed_input_ids_padded if is_packed else input_ids
            sample_mask = (
                packed_rope_attention_mask
                if is_packed
                else (attention_mask.bool() if attention_mask is not None and attention_mask.ndim == 2 else None)
            )
            image_embeds, image_embeds_multiscale = self._encode_and_trim_visual(
                pixel_values,
                image_grid_thw,
                encoder_fn=self.get_image_features,
                token_id=self.image_token_id,
                feature_name="Image features",
                grid_name="image_grid_thw",
                input_name="pixel_values",
                item_counts=multimodal_num_items.get("image_grid_thw") if multimodal_num_items else None,
                sample_tokens=sample_tokens,
                sample_mask=sample_mask,
                target_device=inputs_embeds_bsh.device,
                target_dtype=inputs_embeds_bsh.dtype,
            )
            video_embeds, video_embeds_multiscale = self._encode_and_trim_visual(
                pixel_values_videos,
                video_grid_thw,
                encoder_fn=self.get_video_features,
                token_id=self.video_token_id,
                feature_name="Video features",
                grid_name="video_grid_thw",
                input_name="pixel_values_videos",
                item_counts=multimodal_num_items.get("video_grid_thw") if multimodal_num_items else None,
                sample_tokens=sample_tokens,
                sample_mask=sample_mask,
                target_device=inputs_embeds_bsh.device,
                target_dtype=inputs_embeds_bsh.dtype,
            )

            image_mask, video_mask = self._get_placeholder_mask(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds_bsh,
                image_features=image_embeds,
                video_features=video_embeds,
                image_token_id=self.image_token_id,
                video_token_id=self.video_token_id,
            )

            if image_embeds is not None:
                inputs_embeds_bsh.masked_scatter_(image_mask.unsqueeze(-1).expand_as(inputs_embeds_bsh), image_embeds)
                visual_pos_masks = image_mask
                deepstack_visual_embeds = image_embeds_multiscale

            if video_embeds is not None:
                inputs_embeds_bsh.masked_scatter_(video_mask.unsqueeze(-1).expand_as(inputs_embeds_bsh), video_embeds)
                if deepstack_visual_embeds is None:
                    visual_pos_masks = video_mask
                    deepstack_visual_embeds = video_embeds_multiscale
                else:
                    assert visual_pos_masks is not None
                    visual_pos_masks = visual_pos_masks | video_mask
                    num_visual_tokens = int(visual_pos_masks.sum().item())
                    image_mask_joint = image_mask[visual_pos_masks]
                    video_mask_joint = video_mask[visual_pos_masks]
                    joint_embeds: list[torch.Tensor] = []
                    for image_embed, video_embed in zip(
                        deepstack_visual_embeds, video_embeds_multiscale, strict=True
                    ):
                        embed_joint = image_embed.new_zeros((num_visual_tokens, image_embed.shape[-1]))
                        embed_joint[image_mask_joint] = image_embed
                        embed_joint[video_mask_joint] = video_embed
                        joint_embeds.append(embed_joint)
                    deepstack_visual_embeds = joint_embeds

        if input_features is not None:
            audio_count_input_ids = packed_input_ids_padded if is_packed else input_ids
            expected_audio_token_counts = _count_audio_token_runs(
                audio_count_input_ids,
                self.audio_token_id,
                attention_mask=packed_rope_attention_mask if is_packed else None,
            )
            audio_embeds = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
                expected_audio_token_counts=expected_audio_token_counts,
            )
            audio_embeds = audio_embeds.to(inputs_embeds_bsh.device, inputs_embeds_bsh.dtype)
            audio_mask = input_ids == self.audio_token_id
            if inputs_embeds_bsh[audio_mask].numel() != audio_embeds.numel():
                raise ValueError(
                    f"Audio features and audio tokens do not match: tokens={audio_mask.sum()}, "
                    f"features={audio_embeds.shape[0]}"
                )
            inputs_embeds_bsh.masked_scatter_(audio_mask.unsqueeze(-1).expand_as(inputs_embeds_bsh), audio_embeds)

        return inputs_embeds_bsh.transpose(0, 1).contiguous(), visual_pos_masks, deepstack_visual_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        inference_params: InferenceParams | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        extra_block_kwargs: dict | None = None,
        runtime_gather_output: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        video_second_per_grid: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
        cp_full_input_ids: torch.Tensor | None = None,
        cp_full_lengths: list[int] | None = None,
        cp_local_indices: torch.Tensor | None = None,
        multimodal_num_items: dict[str, list[int]] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if inference_params is not None:
            raise NotImplementedError("Qwen3-Omni Megatron inference is not implemented yet.")

        cp_size = self.pg_collection.cp.size() if self.pg_collection is not None else 1
        cp_rank = self.pg_collection.cp.rank() if self.pg_collection is not None else 0
        tp_size = self.pg_collection.tp.size() if self.pg_collection is not None else 1
        tp_rank = self.pg_collection.tp.rank() if self.pg_collection is not None else 0
        is_packed = packed_seq_params is not None
        packed_cp = is_packed and cp_size > 1
        local_input_ids = input_ids
        if packed_cp:
            if cp_full_input_ids is None or cp_full_lengths is None or cp_local_indices is None:
                raise ValueError("Packed Omni CP requires full tokens, real sample lengths, and local row indices")
            input_ids = cp_full_input_ids

        packed_real_cu_seqlens = None
        packed_input_ids_padded = None
        packed_rope_attention_mask = None
        if packed_cp:
            packed_input_ids_padded, packed_rope_attention_mask = unpack_cp_full_tokens(
                input_ids, packed_seq_params.cu_seqlens_q, cp_full_lengths
            )
        elif is_packed:
            packed_real_cu_seqlens = get_real_cu_seqlens_q(packed_seq_params).to(input_ids.device)
            packed_input_ids_padded = packed_thd_to_padded_batch(input_ids, packed_real_cu_seqlens, pad_value=0)
            packed_rope_attention_mask = packed_attention_mask(
                packed_real_cu_seqlens,
                max_seq_len=packed_input_ids_padded.shape[1],
            ).to(input_ids.device)

        if audio_feature_lengths is None and feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)

        if position_ids is None:
            has_multimodal_inputs = any(
                value is not None
                for value in (pixel_values, pixel_values_videos, input_features, audio_feature_lengths)
            )
            rope_input_ids = packed_input_ids_padded if is_packed else input_ids
            rope_attention_mask = packed_rope_attention_mask if is_packed else attention_mask
            if not has_multimodal_inputs:
                position_ids = _build_text_only_mrope_position_ids(rope_input_ids, rope_attention_mask)
            else:
                image_counts = multimodal_num_items.get("image_grid_thw") if multimodal_num_items else None
                video_counts = multimodal_num_items.get("video_grid_thw") if multimodal_num_items else None
                audio_counts = (
                    multimodal_num_items.get("input_features") or multimodal_num_items.get("feature_attention_mask")
                    if multimodal_num_items
                    else None
                )
                position_ids, _ = get_rope_index(
                    self.config.spatial_merge_size,
                    self.image_token_id,
                    self.video_token_id,
                    self.audio_token_id,
                    self.vision_start_token_id,
                    self.audio_start_token_id,
                    self.position_id_per_seconds,
                    rope_input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grids=video_second_per_grid,
                    attention_mask=rope_attention_mask,
                    audio_seqlens=audio_feature_lengths,
                    image_counts=image_counts,
                    video_counts=video_counts,
                    audio_counts=audio_counts,
                )
            if packed_cp:
                position_ids = pack_cp_position_ids(
                    position_ids, packed_seq_params.cu_seqlens_q, cp_full_lengths, input_ids.shape[1]
                )
            elif is_packed:
                position_ids = pack_padded_batch_position_ids(
                    position_ids,
                    packed_real_cu_seqlens,
                    total_length=input_ids.shape[1],
                )

        if self.pre_process:
            combined_embeddings = self.language_model.embedding(
                input_ids=input_ids,
                position_ids=None,
            ).clone()

            combined_embeddings, visual_pos_masks, deepstack_visual_embeds = self._merge_multimodal_features(
                combined_embeddings,
                input_ids=input_ids,
                is_packed=is_packed,
                packed_input_ids_padded=packed_input_ids_padded,
                packed_rope_attention_mask=packed_rope_attention_mask,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
                multimodal_num_items=multimodal_num_items,
            )

            if packed_cp:
                combined_embeddings = combined_embeddings.index_select(0, cp_local_indices)
                visual_pos_masks, deepstack_visual_embeds = select_packed_cp_deepstack(
                    visual_pos_masks, deepstack_visual_embeds, cp_local_indices
                )
            elif combined_embeddings is not None and cp_size > 1 and packed_seq_params is None:
                combined_embeddings = split_data_for_context_parallel(combined_embeddings, cp_size, cp_rank)

            sp_pad_len = 0
            if self.config.sequence_parallel:
                seq_len = combined_embeddings.shape[0]
                sp_pad_len = (tp_size - seq_len % tp_size) % tp_size
                if is_packed and sp_pad_len > 0:
                    raise ValueError(
                        f"Packed Qwen3-Omni sequence length must be TP-aligned before sequence parallel scatter: "
                        f"seq_len={seq_len}, tp_size={tp_size}"
                    )
                elif sp_pad_len > 0:
                    combined_embeddings = torch.nn.functional.pad(combined_embeddings, (0, 0, 0, 0, 0, sp_pad_len))
                    if visual_pos_masks is not None:
                        visual_pos_masks = torch.nn.functional.pad(visual_pos_masks, (0, sp_pad_len), value=False)
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
                combined_embeddings = combined_embeddings.contiguous()
        else:
            combined_embeddings = None
            visual_pos_masks = None
            deepstack_visual_embeds = None
            sp_pad_len = 0

        if sp_pad_len > 0 and position_ids is not None:
            position_ids = torch.nn.functional.pad(position_ids, (0, sp_pad_len), mode="replicate")

        if self.config.sequence_parallel or cp_size > 1:
            visual_pos_masks, deepstack_visual_embeds = shard_deepstack_for_model_parallel(
                visual_pos_masks,
                deepstack_visual_embeds,
                tp_size=tp_size,
                tp_rank=tp_rank,
                cp_size=1 if packed_cp else cp_size,
                cp_rank=0 if packed_cp else cp_rank,
                sequence_parallel=self.config.sequence_parallel,
            )

        with torch.profiler.record_function("omni.llm_forward"):
            return self.language_model(
                input_ids=None if combined_embeddings is not None else local_input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=combined_embeddings,
                labels=labels,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                runtime_gather_output=runtime_gather_output,
                extra_block_kwargs=extra_block_kwargs,
                loss_mask=loss_mask,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )
