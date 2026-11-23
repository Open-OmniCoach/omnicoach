"""Keep HF's audio convolution batch size independent of the final chunk count."""

import torch.nn.functional as F
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    chunk_and_pad_features,
)


class Qwen3OmniAudioEncoder(Qwen3OmniMoeAudioEncoder):
    def forward(self, input_features=None, feature_lens=None, **kwargs):
        padded, lengths = chunk_and_pad_features(input_features, feature_lens, self.n_window, kwargs=kwargs)
        extra = -padded.shape[0] % self.conv_chunksize
        if extra:
            padded = F.pad(padded, (0, 0, 0, 0, 0, extra))
            # HF's valid-index mask excludes every output from these empty chunks.
            lengths = F.pad(lengths, (0, extra))
        return super().forward(
            input_features,
            feature_lens,
            padded_feature=padded,
            chunk_lengths=lengths,
            **kwargs,
        )
