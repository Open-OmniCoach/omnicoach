"""Patch for ``slime.backends.megatron_utils.actor.MegatronTrainRayActor``.

1. ``fill_routing_replay`` collects active ``RoutingReplay`` instances from
   ``self.model`` in exact vp_stage/layer order (immune to stale routers left in
   the global ``RoutingReplay.all_routing_replays`` registry) while preserving
   upstream Slime's disk-backed ``TensorRef`` lazy prefetching
   (``RoutedExpertsMicrobatchPrefetcher``) and per-microbatch RSS release.
2. ``_get_rollout_data`` wraps upstream's transfer loop cleanly: when
   ``SLIME_OMNI_LAZY_MULTIMODAL_H2D=1``, it temporarily detaches
   ``multimodal_train_inputs`` from the unpacked batch before delegating to
   upstream and re-attaches the CPU tensors afterward so ``get_batch`` packs and
   transfers them per microbatch; it also prepares optional turn-level fields
   (``token_advantages``, ``turn_ids``, ``turn_weights``).
"""

from functools import wraps

import torch
from slime.backends.megatron_utils.cp_utils import prepare_routed_experts_for_routing_replay
from slime.data.tensor import TensorRef

_applied = False


def _prepare_turn_level_fields(rollout_data, device, slice_fn):
    """Move turn-level fields to device with matching CP layout.

    - ``token_advantages``: CP-sliced (matching ``rollout_log_probs`` / ``advantages``), float32.
    - ``turn_ids``: full response length (matching ``loss_masks``), int64.
    - ``turn_weights``: full response length (matching ``loss_masks``), float32.
    """
    if "token_advantages" in rollout_data and rollout_data["token_advantages"] is not None:
        rollout_data["token_advantages"] = [
            slice_fn(
                torch.as_tensor(adv, dtype=torch.float32),
                total_length,
                response_length,
            ).to(device=device, dtype=torch.float32, non_blocking=True)
            for adv, total_length, response_length in zip(
                rollout_data["token_advantages"],
                rollout_data["total_lengths"],
                rollout_data["response_lengths"],
                strict=True,
            )
        ]
    if "turn_ids" in rollout_data and rollout_data["turn_ids"] is not None:
        rollout_data["turn_ids"] = [
            torch.as_tensor(turn_ids, dtype=torch.long).to(
                device=device, dtype=torch.long, non_blocking=True
            )
            for turn_ids in rollout_data["turn_ids"]
        ]
    if "turn_weights" in rollout_data and rollout_data["turn_weights"] is not None:
        rollout_data["turn_weights"] = [
            torch.as_tensor(weights, dtype=torch.float32).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            for weights in rollout_data["turn_weights"]
        ]
    return rollout_data


def _make_get_rollout_data(original):
    @wraps(original)
    def get_rollout_data(self, rollout_data_ref):
        from slime.backends.megatron_utils import actor as mod

        from slime_ext.backends.megatron.data import _lazy_multimodal_h2d_enabled

        device = mod.accelerator.current_device()
        if not _lazy_multimodal_h2d_enabled():
            rollout_data = original(self, rollout_data_ref)
            return _prepare_turn_level_fields(rollout_data, device, mod.slice_log_prob_with_cp)

        sentinel = object()
        cpu_multimodal = sentinel
        orig_process = mod.process_rollout_data

        def _defer_multimodal(*args, **kwargs):
            nonlocal cpu_multimodal
            data = orig_process(*args, **kwargs)
            if "multimodal_train_inputs" in data:
                cpu_multimodal = data.pop("multimodal_train_inputs")
            return data

        mod.process_rollout_data = _defer_multimodal
        try:
            rollout_data = original(self, rollout_data_ref)
        finally:
            mod.process_rollout_data = orig_process
        if cpu_multimodal is not sentinel:
            rollout_data["multimodal_train_inputs"] = cpu_multimodal
        return _prepare_turn_level_fields(rollout_data, device, mod.slice_log_prob_with_cp)

    return get_rollout_data


def _moe_layer_ids(config, offset, num_layers_to_build):
    layer_ids = []
    for layer_id in range(offset, offset + num_layers_to_build):
        if isinstance(config.moe_layer_freq, int):
            if layer_id % config.moe_layer_freq != 0:
                continue
        elif isinstance(config.moe_layer_freq, list):
            assert len(config.moe_layer_freq) == config.num_layers
            if config.moe_layer_freq[layer_id] == 0:
                continue
        layer_ids.append(layer_id)
    return layer_ids


def _collect_active_routing_replays(models):
    from megatron.core.transformer.transformer_block import get_num_layers_to_build
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    active_replays = []
    for vp_stage, model in enumerate(models):
        config = model.module.config
        num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        expected_layer_ids = _moe_layer_ids(config, offset, num_layers_to_build)

        # Collect this model's routers in module-registration order.
        model_replays = [
            module.routing_replay
            for _, module in model.module.named_modules()
            if hasattr(module, "routing_replay")
        ]
        assert len(model_replays) == len(expected_layer_ids), (
            f"vp_stage={vp_stage}: {len(model_replays)} routers in model but "
            f"{len(expected_layer_ids)} MoE layers expected"
        )
        for layer_id, replay in zip(expected_layer_ids, model_replays, strict=True):
            active_replays.append((layer_id, replay))
    return active_replays


def fill_routing_replay(self, data_iterator, num_microbatches, rollout_data):
    """Record rollout routed experts into active model routers with R3 prefetching."""
    if "rollout_routed_experts" not in rollout_data:
        raise ValueError(
            "rollout_routed_experts is required in rollout_data when use_rollout_routing_replay is set."
        )

    from slime.utils.routed_experts import (
        RoutedExpertsLayerRef,
        RoutedExpertsMicrobatch,
        RoutedExpertsMicrobatchPrefetcher,
    )
    from slime.utils.routing_replay import RoutingReplay

    active_replays = _collect_active_routing_replays(self.model)

    for iterator in data_iterator:
        iterator.reset()

    replay_source = rollout_data["rollout_routed_experts"]
    disk_prefetcher = None
    prepare_kwargs = {
        "num_experts": self.args.num_experts,
        "data_pad_size_multiplier": self.args.data_pad_size_multiplier,
        "sequence_parallel": self.args.sequence_parallel,
        "allgather_cp": self.args.allgather_cp,
    }

    for _microbatch_idx in range(sum(num_microbatches)):
        iterator = data_iterator[0]
        batch_indices = iterator.micro_batch_indices[iterator.offset]
        batch = iterator.get_next(["rollout_routed_experts", "tokens"])
        values = batch["rollout_routed_experts"]

        disk_backed = [isinstance(value, TensorRef) for value in values]
        if any(disk_backed) and not all(disk_backed):
            raise ValueError("A routing replay microbatch cannot mix disk-backed and resident route tensors.")

        if active_replays and all(disk_backed):
            if disk_prefetcher is None:
                disk_prefetcher = RoutedExpertsMicrobatchPrefetcher(
                    self.args.routing_replay_prefetch_microbatches
                )
            source = RoutedExpertsMicrobatch(
                values,
                batch["tokens"],
                consumer_count=len(active_replays),
                prepare_kwargs=prepare_kwargs,
            )
            disk_prefetcher.add(source)
            for layer_id, replay in active_replays:
                replay.record(RoutedExpertsLayerRef(source, layer_id))
        elif active_replays:
            rollout_routed_experts = prepare_routed_experts_for_routing_replay(
                values,
                batch["tokens"],
                **prepare_kwargs,
            )
            for layer_id, replay in active_replays:
                replay.record(rollout_routed_experts[:, layer_id])

        # Drop manager-owned references as soon as this microbatch has been
        # registered, bounding actor RSS during setup.
        for sample_idx in batch_indices:
            replay_source[sample_idx] = None

    if disk_prefetcher is not None:
        disk_prefetcher.start()
        RoutingReplay.register_lazy_resource(disk_prefetcher)

    del rollout_data["rollout_routed_experts"]

    for iterator in data_iterator:
        iterator.reset()


def patch():
    global _applied
    if _applied:
        return

    import slime.backends.megatron_utils.actor as mod

    mod.MegatronTrainRayActor.fill_routing_replay = fill_routing_replay
    mod.MegatronTrainRayActor._get_rollout_data = _make_get_rollout_data(
        mod.MegatronTrainRayActor._get_rollout_data
    )
    _applied = True
