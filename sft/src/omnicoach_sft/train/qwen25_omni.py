"""Explicit full-audio, dataset-owned-system policy for Qwen2.5 trace SFT.

Call after Swift constructs the template. This does not register global patches
or change Qwen3 behavior; training and inference must opt into the same policy.
"""
from copy import deepcopy

from torch.nn import functional as F
from transformers import WhisperFeatureExtractor


class FullAudioWhisperFeatureExtractor(WhisperFeatureExtractor):
    """Preserve full waveforms; sequence-length rejection belongs to the template."""

    def __call__(self, *args, **kwargs):
        kwargs['truncation'] = False
        return super().__call__(*args, **kwargs)

    def to_dict(self):
        # Checkpoints must remain loadable by the standard HF processor. The
        # full-audio runtime policy is reapplied by the training/inference
        # adapter rather than publishing an unregistered extractor class.
        payload = super().to_dict()
        payload['feature_extractor_type'] = 'WhisperFeatureExtractor'
        return payload


class FullAudioBatchCollator:
    """Pad Qwen2.5 audio features across rows without changing valid frames."""

    def __init__(self, template):
        self.template = template
        # Keep the class function: a pickled bound method would resolve back to
        # this installed wrapper and recurse when DataLoader restores it.
        self.delegate = type(template)._data_collator_mm_data

    def __call__(self, batch):
        audio_rows = [row for row in batch if row.get('input_features') is not None]
        if not audio_rows:
            return self.delegate(self.template, batch)
        max_length = max(row['input_features'].shape[-1] for row in audio_rows)
        padded_batch = [dict(row) for row in batch]
        for row in padded_batch:
            features = row.get('input_features')
            if features is None:
                continue
            mask = row['feature_attention_mask']
            if features.shape[0] != mask.shape[0] or features.shape[-1] != mask.shape[-1]:
                raise ValueError('Qwen2.5 audio features and attention mask must have matching lengths')
            pad = max_length - features.shape[-1]
            if pad:
                row['input_features'] = F.pad(features, (0, pad))
                row['feature_attention_mask'] = F.pad(mask, (0, pad))
        return self.delegate(self.template, padded_batch)


def configure_qwen25_trace_template(template):
    if getattr(template, 'version', None) != 'omni_v2_5':
        raise ValueError('full-audio trace adapter requires Qwen2.5-Omni')
    template.template_meta = deepcopy(template.template_meta)
    template.template_meta.default_system = None
    extractor = template.processor.feature_extractor
    template.processor.feature_extractor = FullAudioWhisperFeatureExtractor.from_dict(extractor.to_dict())
    if not isinstance(template._data_collator_mm_data, FullAudioBatchCollator):
        template._data_collator_mm_data = FullAudioBatchCollator(template)
