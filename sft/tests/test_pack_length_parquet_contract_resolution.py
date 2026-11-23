from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from omnicoach_sft.media.image_tokens import QWEN_IMAGE_TOKEN_COUNT_BASIS
from omnicoach_sft.train.lengths import ContractLengthStats, _load_contract_stats


def test_contract_stats_resolve_from_turns_part_and_sample_syntax(tmp_path: Path) -> None:
    artifact = tmp_path / "agent_artifact"
    turns_dir = artifact / "turns"
    turns_dir.mkdir(parents=True)
    part = turns_dir / "part-00000.parquet"
    Dataset.from_dict({"value": [1]}).to_parquet(part)

    contract = {
        "schema_version": "agent_turn_shared_media_v1",
        "table_layout": "agent_turn_shared_media_v1",
        "purpose": "agentic_turn_sft",
        "protocol": "native-tool-v3-seconds",
        "source_id": "agent_fixture",
        "pack_length_stats": {
            "image_max_pixels": 147_456,
            "image_token_count_basis": QWEN_IMAGE_TOKEN_COUNT_BASIS,
            "image_token_count_column": "omni_image_num_tokens",
            "audio_sample_rate": 16_000,
            "audio_sample_count_basis": "resampled_16khz",
        },
    }
    contract_path = artifact / "dataset_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    import hashlib

    expected = ContractLengthStats(
        contract_hash=hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        source_id="agent_fixture",
        image_max_pixels=147_456,
        audio_sample_rate=16_000,
        is_agent=True,
        has_exact_media_lengths=True,
    )
    assert _load_contract_stats(artifact) == expected
    assert _load_contract_stats(part) == expected
    assert _load_contract_stats(f"{part}#1") == expected
