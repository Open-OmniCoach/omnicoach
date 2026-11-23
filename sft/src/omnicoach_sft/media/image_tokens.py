"""Image smart-resize and Qwen vision token-count estimation helpers."""

from __future__ import annotations

import math


def round_by_factor(number: float, factor: int) -> int:
    return int(round(number / factor) * factor)


def ceil_by_factor(number: float, factor: int) -> int:
    return int(math.ceil(number / factor) * factor)


def floor_by_factor(number: float, factor: int) -> int:
    return int(math.floor(number / factor) * factor)


def smart_resize(*, width: int, height: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    factor = max(1, int(factor))
    min_pixels = max(1, int(min_pixels))
    max_pixels = max(min_pixels, int(max_pixels))
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = max(factor, ceil_by_factor(height * beta, factor))
        w_bar = max(factor, ceil_by_factor(width * beta, factor))
    while h_bar * w_bar > max_pixels and (h_bar > factor or w_bar > factor):
        if w_bar >= h_bar and w_bar > factor:
            w_bar -= factor
        elif h_bar > factor:
            h_bar -= factor
        else:
            break
    return w_bar, h_bar



QWEN_IMAGE_PATCH_SIZE = 16
QWEN_IMAGE_SPATIAL_MERGE_SIZE = 2
QWEN_IMAGE_FACTOR = QWEN_IMAGE_PATCH_SIZE * QWEN_IMAGE_SPATIAL_MERGE_SIZE
QWEN_IMAGE_MIN_TOKENS = 4
QWEN_IMAGE_MAX_TOKENS = 16384
QWEN_IMAGE_WRAPPER_TOKENS = 2
QWEN_IMAGE_TOKEN_COUNT_BASIS = "qwen3_omni_v3_patch16_merge2_plus_vision_wrappers"


def webp_dimensions(payload: bytes, *, context: str) -> tuple[int, int]:
    """Read dimensions from the fixed-size header of a RIFF WebP payload."""

    if len(payload) < 30 or payload[:4] != b"RIFF" or payload[8:12] != b"WEBP":
        raise ValueError(f"{context}: image is not a RIFF WebP payload")
    chunk = payload[12:16]
    if chunk == b"VP8X":
        width = int.from_bytes(payload[24:27], "little") + 1
        height = int.from_bytes(payload[27:30], "little") + 1
    elif chunk == b"VP8L":
        if payload[20] != 0x2F:
            raise ValueError(f"{context}: invalid VP8L signature")
        bits = int.from_bytes(payload[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
    elif chunk == b"VP8 ":
        if payload[23:26] != b"\x9d\x01\x2a":
            raise ValueError(f"{context}: invalid VP8 key-frame signature")
        width = int.from_bytes(payload[26:28], "little") & 0x3FFF
        height = int.from_bytes(payload[28:30], "little") & 0x3FFF
    else:
        raise ValueError(f"{context}: unsupported WebP chunk: {chunk!r}")
    if width <= 0 or height <= 0:
        raise ValueError(f"{context}: invalid WebP dimensions: {width}x{height}")
    return width, height


def qwen_image_token_contribution(width: int, height: int) -> int:
    """Return merged vision-grid tokens plus the stock two-token wrapper."""

    if width <= 0 or height <= 0:
        raise ValueError(f"image dimensions must be positive, got {width}x{height}")
    if max(width, height) / min(width, height) > 200:
        raise ValueError(f"absolute image aspect ratio must be <= 200, got {width}x{height}")
    factor = QWEN_IMAGE_FACTOR
    resized_width, resized_height = smart_resize(
        width=width,
        height=height,
        factor=factor,
        min_pixels=QWEN_IMAGE_MIN_TOKENS * factor**2,
        max_pixels=QWEN_IMAGE_MAX_TOKENS * factor**2,
    )
    merged_grid_tokens = (resized_width // factor) * (resized_height // factor)
    return merged_grid_tokens + QWEN_IMAGE_WRAPPER_TOKENS


def qwen_webp_token_contribution(payload: bytes, *, context: str) -> int:
    width, height = webp_dimensions(payload[:30], context=context)
    return qwen_image_token_contribution(width, height)
