"""HF Omni vision with a batch-independent order for bilinear interpolation.

The forward follows Transformers 5.12.1. Only the four-corner reduction differs:
Qwen's explicit left-to-right sum (Transformers v4.57.1, modeling_qwen3_omni_moe.py)
avoids torch.sum(0) choosing a different reduction order for larger packed batches.
Interpolation indices, FP32 weights, model parameters and attention remain HF's.
"""

import torch
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    BaseModelOutputWithDeepstackFeatures,
    Qwen3OmniMoeVisionEncoder,
)
from transformers.vision_utils import (
    get_vision_bilinear_indices_and_weights,
    get_vision_cu_seqlens,
    get_vision_position_ids,
)


class Qwen3OmniVisionEncoder(Qwen3OmniMoeVisionEncoder):
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
        indices, weights = get_vision_bilinear_indices_and_weights(
            grid_thw, self.num_grid_per_side, self.spatial_merge_size, kwargs=kwargs
        )
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
        cu_seqlens = get_vision_cu_seqlens(grid_thw, kwargs=kwargs)
        hidden_states = self.patch_embed(hidden_states)
        corners = self.pos_embed(indices) * weights[:, :, None]
        pos_embeds = ((corners[0] + corners[1]) + corners[2]) + corners[3]
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
        del corners, pos_embeds
        rotary_pos_emb = self.rotary_pos_emb(position_ids)
        hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)
        rotary_pos_emb = rotary_pos_emb.reshape(hidden_states.shape[0], -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        deepstack_features = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings, **kwargs
            )
            if layer_num in self.deepstack_visual_indexes:
                merger = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)]
                deepstack_features.append(merger(hidden_states))
        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=self.merger(hidden_states),
            deepstack_features=deepstack_features,
        )
