#!/usr/bin/env python3
"""Convert Qwen3-Omni Thinker decoder weights to blockwise FP8.

``decoder`` quantizes all Thinker decoder linear layers (attention projections
and MoE experts); ``experts_only`` keeps attention projections in BF16 while
quantizing only the MoE expert weights.  Multimedia towers, embeddings,
normalizations, LM head, and routers remain BF16 in both scopes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import safetensors
import safetensors.torch


def _load_slime_converter(path: Path):
    spec = importlib.util.spec_from_file_location("slime_fp8_converter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Slime converter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _should_quantize(key: str, weight, scope: str) -> bool:
    if not (
        key.startswith("thinker.model.layers.")
        and weight.ndim == 2
        and key.endswith(".weight")
    ):
        return False
    if scope == "experts_only":
        return ".mlp.experts." in key
    return not any(
        token in key
        for token in (
            "layernorm",
            "embed",
            "router",
            "mlp.gate.",
            "norm",
            "lm_head",
            "eh_proj",
            "weights_proj",
            "conv1d",
            "A_log",
            "dt_bias",
            "in_proj_a",
            "in_proj_b",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--slime-converter", type=Path, required=True)
    parser.add_argument(
        "--scope", choices=("decoder", "experts_only"), default="decoder"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()

    converter = _load_slime_converter(args.slime_converter)

    def process_file(
        input_path,
        output_path,
        filename,
        strategy,
        block_size,
        result_collector,
    ):
        if not filename.endswith(".safetensors"):
            return
        weights = {}
        quantized = {}
        with safetensors.safe_open(
            os.path.join(input_path, filename), framework="pt", device=args.device
        ) as stream:
            for key in stream.keys():
                weights[key] = stream.get_tensor(key)

        modules_to_not_convert = []
        for key, weight in weights.items():
            if _should_quantize(key, weight, args.scope):
                qweight, scale = converter.quant_fp8(weight, strategy, block_size)
                quantized[key] = qweight
                quantized[key.replace(".weight", ".weight_scale_inv")] = scale
            else:
                modules_to_not_convert.append(key.removesuffix(".weight"))
                quantized[key] = weight

        safetensors.torch.save_file(
            quantized,
            os.path.join(output_path, filename),
            metadata={"format": "pt"},
        )
        result_collector.add_result(filename, quantized, modules_to_not_convert)

    converter.process_file = process_file
    converter.convert_fp8(
        str(args.model_dir),
        str(args.save_dir),
        "block",
        [128, 128],
        args.max_workers,
        None,
    )

    config_path = args.save_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quantization_config = config["quantization_config"]
    quantization_config.pop("modules_to_not_convert", None)
    ignored_layers = [
        "thinker.visual",
        "thinker.audio_tower",
        "code2wav",
        "lm_head",
        "input_layernorm",
        "post_attention_layernorm",
        "mlp.gate",
        "_norm",
        "embed_tokens",
        "codec_embedding",
        "visual",
        "blocks",
        "merger",
        "deepstack_merger_list",
        "merger_list",
        "pos_embed",
    ]
    if args.scope == "experts_only":
        ignored_layers.append("self_attn")
    quantization_config["ignored_layers"] = ignored_layers
    quantization_config["slime_ext_qwen3omni_scope"] = args.scope
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
