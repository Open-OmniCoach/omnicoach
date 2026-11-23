"""Override for ``slime.backends.megatron_utils.data.get_batch``.

Extends upstream Slime's THD microbatch packer with:
- Audio ``PAD_RULES`` padding before flattening,
- Per-sample ``multimodal_num_items`` aligned across *every* sample in the
  microbatch (emitting ``0`` for samples without a modality) and passed inside
  ``multimodal_train_inputs`` so ``Qwen3OmniThinkerModel`` can align and trim
  visual features per sample without cross-sample bleed,
- ``PackedSeqParamsWithCPUOffsets`` and host-side ``max_seqlen`` calculation
  (eliminating per-microbatch CUDA sync),
- ``SLIME_OMNI_LAZY_MULTIMODAL_H2D=1`` safe CPU staging for list-valued and
  heterogeneous-shape multimodal fields before single microbatch H2D transfer,
- ``real_cu_seqlens``, packed CP full-token metadata, and ``full_response_row_masks``.
"""

import os

import torch

from slime_ext.backends.megatron.packed import attach_real_cu_seqlens_q

_applied = False

PAD_RULES = {
    # input_features is shaped [num_audio, feature_dim, time].
    "input_features": dict(pad_dim=-1, padding_value=0.0),
    # feature_attention_mask is shaped [num_audio, time].
    "feature_attention_mask": dict(pad_dim=-1, padding_value=0),
}


def _lazy_multimodal_h2d_enabled():
    value = os.getenv("SLIME_OMNI_LAZY_MULTIMODAL_H2D", "")
    return value.strip().lower() not in {"", "0", "false", "no", "off", "none"}


def move_multimodal_to_device(multimodal_data, device):
    """Move one packed multimodal microbatch while preserving metadata values."""
    moved = {}
    for key, value in multimodal_data.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device=device, non_blocking=True)
        elif isinstance(value, list) and value and all(isinstance(item, torch.Tensor) for item in value):
            moved[key] = [item.to(device=device, non_blocking=True) for item in value]
        else:
            moved[key] = value
    return moved


def pad_and_flatten(tensor_list, pad_dim=-1, padding_value=0):
    if len(tensor_list) == 1:
        tensor = tensor_list[0]
        return tensor, [tensor.size(0)]

    if pad_dim < 0:
        pad_dim = tensor_list[0].dim() + pad_dim
    max_len = max(tensor.size(pad_dim) for tensor in tensor_list)

    padded_list = []
    for tensor in tensor_list:
        if tensor.size(pad_dim) < max_len:
            pad_shape = list(tensor.shape)
            pad_shape[pad_dim] = max_len - tensor.size(pad_dim)
            pad = tensor.new_full(pad_shape, padding_value)
            tensor = torch.cat([tensor, pad], dim=pad_dim)
        padded_list.append(tensor)

    return torch.cat(padded_list, dim=0), [tensor.size(0) for tensor in padded_list]


def _runtime_response_only_lm_head_enabled():
    from megatron.training import get_args

    return bool(getattr(get_args(), "response_only_lm_head", False))


def _packed_cp_model_inputs(tokens, cp_size, tail_pad, slice_with_cp):
    """Keep full media positions while following Slime's per-sample CP layout."""
    full_tokens, local_indices = [], []
    offset = 0
    lengths = [t.numel() for t in tokens]
    for token_ids, length in zip(tokens, lengths, strict=True):
        padded_length = (length + 2 * cp_size - 1) // (2 * cp_size) * (2 * cp_size)
        full_tokens.append(torch.nn.functional.pad(token_ids, (0, padded_length - length), value=0))
        indices = torch.arange(offset, offset + padded_length, device=token_ids.device)
        local_indices.append(slice_with_cp(indices, 0))
        offset += padded_length

    if tail_pad:
        full_tokens.append(tokens[0].new_zeros(tail_pad * cp_size))
        indices = torch.arange(offset, offset + tail_pad * cp_size, device=tokens[0].device)
        local_indices.append(slice_with_cp(indices, 0))

    return {
        "cp_full_input_ids": torch.cat(full_tokens).unsqueeze(0),
        "cp_full_lengths": lengths,
        "cp_local_indices": torch.cat(local_indices),
    }


def _make_get_batch(
    mpu,
    F,
    PackedSeqParams,
    slice_with_cp,
    response_only_lm_head_enabled=lambda: False,
):
    supports_cpu_offsets = (
        hasattr(PackedSeqParams, "__dataclass_fields__")
        and "cu_seqlens_q_cpu" in PackedSeqParams.__dataclass_fields__
    )

    def get_batch(data_iterator, keys, pad_multiplier=128, allgather_cp=False):
        """CP-ready micro-batch with packed sequence params (see upstream docstring)."""
        assert "tokens" in keys
        batch = data_iterator.get_next(keys)

        tokens = batch["tokens"]
        pad_token_id = 0
        pad_size = mpu.get_tensor_model_parallel_world_size() * pad_multiplier

        batch["unconcat_tokens"] = tokens

        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()

        if not allgather_cp:
            tokens = [slice_with_cp(t, pad_token_id) for t in tokens]
        cu_seqlens_list = [0]
        for tensor in tokens:
            cu_seqlens_list.append(cu_seqlens_list[-1] + tensor.size(0))
        tokens = torch.cat(tokens, dim=0)

        # All-gather packs global tokens; zigzag CP packs each rank's shard.
        real_cu_seqlens_list = (
            [offset * cp_size for offset in cu_seqlens_list]
            if not allgather_cp
            else list(cu_seqlens_list)
        )
        real_cu_seqlens = torch.tensor(real_cu_seqlens_list, dtype=torch.int, device=tokens.device)

        packing_size = pad_size * cp_size if allgather_cp else pad_size
        pad = (packing_size - tokens.size(0) % packing_size) % packing_size
        if pad != 0:
            tokens = F.pad(tokens, (0, pad), value=pad_token_id)
            cu_seqlens_list.append(cu_seqlens_list[-1] + pad)

        if not allgather_cp:
            cu_seqlens_list = [offset * cp_size for offset in cu_seqlens_list]
        else:
            tokens = tokens.chunk(cp_size, dim=0)[cp_rank]

        cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int, device=tokens.device)
        # Compute max_seqlen on CPU without synchronizing the GPU.
        max_seqlen = max(
            end - start for start, end in zip(cu_seqlens_list[:-1], cu_seqlens_list[1:], strict=True)
        )
        packed_kwargs = dict(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format="thd",
        )
        if supports_cpu_offsets:
            packed_kwargs["cu_seqlens_q_cpu"] = tuple(cu_seqlens_list)
        packed_seq_params = PackedSeqParams(**packed_kwargs)
        attach_real_cu_seqlens_q(packed_seq_params, real_cu_seqlens)

        tokens = tokens.unsqueeze(0)

        batch["tokens"] = tokens
        batch["packed_seq_params"] = packed_seq_params

        loss_masks = []
        response_row_masks = []
        for loss_mask, total_length, response_length in zip(
            batch["loss_masks"],
            batch["total_lengths"],
            batch["response_lengths"],
            strict=True,
        ):
            prompt_length = total_length - response_length
            if loss_mask.numel() != response_length:
                raise ValueError(
                    f"loss mask has {loss_mask.numel()} rows for response_length={response_length}"
                )
            response_row_mask = F.pad(
                torch.ones_like(loss_mask),
                (prompt_length - 1, 1),
                value=0,
            )
            loss_mask = F.pad(loss_mask, (prompt_length - 1, 1), value=0)
            if not allgather_cp:
                loss_mask = slice_with_cp(loss_mask, 0)
                response_row_mask = slice_with_cp(response_row_mask, 0)
            loss_masks.append(loss_mask)
            response_row_masks.append(response_row_mask)

        packed_masks = {}
        for name, masks in (("loss_masks", loss_masks), ("response_row_masks", response_row_masks)):
            masks = torch.cat(masks, dim=0)
            if pad != 0:
                masks = F.pad(masks, (0, pad), value=0)
            if allgather_cp:
                masks = masks.chunk(cp_size, dim=0)[cp_rank]
            masks = masks.unsqueeze(0)
            assert masks.shape == tokens.shape, f"{name}.shape: {masks.shape}, tokens.shape: {tokens.shape}"
            packed_masks[name] = masks
        batch["full_response_row_masks"] = packed_masks["response_row_masks"]
        batch["full_loss_masks"] = packed_masks[
            "response_row_masks" if response_only_lm_head_enabled() else "loss_masks"
        ]

        # Multimodal packing: pad unequal-length fields per PAD_RULES before flatten.
        multimodal_train_inputs = batch.get("multimodal_train_inputs", None)
        if multimodal_train_inputs is not None:
            lazy_h2d = _lazy_multimodal_h2d_enabled()
            num_samples = len(multimodal_train_inputs)
            multimodal_data = {}
            multimodal_num_items = {}
            tensor_dict_list = {}
            sample_item_counts = {}

            for sample_idx, mm_input_dict in enumerate(multimodal_train_inputs):
                if mm_input_dict is None:
                    continue
                for key, mm_tensor in mm_input_dict.items():
                    if isinstance(mm_tensor, list):
                        # Under lazy H2D keep staging tensors on CPU so they match
                        # sibling CPU tensors before the single microbatch H2D move.
                        target_device = torch.device("cpu") if lazy_h2d else tokens.device
                        mm_tensor = torch.tensor(mm_tensor, device=target_device)
                    tensor_dict_list.setdefault(key, []).append(mm_tensor)
                    counts = sample_item_counts.setdefault(key, [0] * num_samples)
                    counts[sample_idx] = int(mm_tensor.size(0)) if isinstance(mm_tensor, torch.Tensor) and mm_tensor.ndim > 0 else 1

            for key, tensor_list in tensor_dict_list.items():
                if key in PAD_RULES:
                    multimodal_data[key], _ = pad_and_flatten(tensor_list, **PAD_RULES[key])
                elif len(tensor_list) == 1:
                    multimodal_data[key] = tensor_list[0]
                elif all(t.shape[1:] == tensor_list[0].shape[1:] for t in tensor_list):
                    # Align devices before cat if any sample produced a mismatched staging tensor.
                    base_device = tensor_list[0].device
                    aligned = [t if t.device == base_device else t.to(base_device) for t in tensor_list]
                    multimodal_data[key] = torch.cat(aligned, dim=0)
                else:
                    multimodal_data[key] = tensor_list
                multimodal_num_items[key] = sample_item_counts[key]

            if lazy_h2d:
                multimodal_data = move_multimodal_to_device(multimodal_data, tokens.device)

            if multimodal_num_items:
                multimodal_data["multimodal_num_items"] = multimodal_num_items
            batch["multimodal_train_inputs"] = multimodal_data
            batch["multimodal_num_items"] = multimodal_num_items

        if cp_size > 1 and not allgather_cp:
            multimodal_data = dict(batch.get("multimodal_train_inputs") or {})
            multimodal_data.update(
                _packed_cp_model_inputs(batch["unconcat_tokens"], cp_size, pad, slice_with_cp)
            )
            batch["multimodal_train_inputs"] = multimodal_data

        return batch

    return get_batch


def patch():
    global _applied
    if _applied:
        return
    import sys

    import slime.backends.megatron_utils.data as mod

    packed_cls = getattr(mod, "PackedSeqParamsWithCPUOffsets", mod.PackedSeqParams)
    get_batch = _make_get_batch(
        mod.mpu,
        mod.F,
        packed_cls,
        mod.slice_with_cp,
        _runtime_response_only_lm_head_enabled,
    )
    mod.get_batch = get_batch
    mod.PAD_RULES = PAD_RULES
    mod.pad_and_flatten = pad_and_flatten
    mod.move_multimodal_to_device = move_multimodal_to_device

    # model.py does `from .data import get_batch` at import time, binding the old
    # reference into its own namespace. Rebind it there too if already imported.
    model_mod = sys.modules.get("slime.backends.megatron_utils.model")
    if model_mod is not None and hasattr(model_mod, "get_batch"):
        model_mod.get_batch = get_batch

    _applied = True
