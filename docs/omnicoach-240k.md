# Download and use OmniCoach-240K

[OmniCoach-240K on Hugging Face](https://huggingface.co/datasets/Open-OmniCoach/OmniCoach-240K) hosts the seed SFT trajectories, prebuilt shared-media SFT Parquet, RL tasks, and source-media manifests. Start with the files for your workflow.

| What you need | Hugging Face path or config |
| --- | --- |
| Preview the seed SFT trajectories | `sft_seed` config, `train` split |
| Download seed trajectory JSONL | `sft/raw_data/seed/` (~3.5 GB) |
| Train with the prebuilt SFT Parquet | `sft/parquet/seed/` (16 parts, ~856 GB total) |
| Read the 10,000 RL tasks | `rl/train.parquet` (~1.9 MB) |
| Identify the original videos and audio | `media/` (~3.9 MB) |

Install the Hugging Face CLI for the download commands below with `python -m pip install -U huggingface_hub`.

## Preview seed SFT trajectories

Install `datasets` and stream a row without downloading the full split:

```python
from datasets import load_dataset

traces = load_dataset(
    "Open-OmniCoach/OmniCoach-240K", "sft_seed", split="train", streaming=True
)
row = next(iter(traces))
print(row["trajectory_id"], row["video_source"])
```

To save the raw seed JSONL files, run:

```bash
hf download Open-OmniCoach/OmniCoach-240K --repo-type dataset \
  --include 'sft/raw_data/seed/*.jsonl' \
  --local-dir ./OmniCoach-240K
```

## Download prebuilt SFT Parquet

This example downloads one complete part (~46 GB). Replace `part_01` to select a different part, or use `'sft/parquet/seed/*'` to download all 16 parts.

```bash
hf download Open-OmniCoach/OmniCoach-240K --repo-type dataset \
  --include 'sft/parquet/seed/part_01/*' \
  --local-dir ./OmniCoach-240K
```

Keep `dataset_contract.json`, `turns/`, and `media/` together in each `part_XX/` directory. The turn rows reference encoded frames and audio in that part's media table. The prebuilt seed parts already contain these training inputs.

After setting up the [SFT environment](../sft/README.md#setup), validate the part with its local absolute path:

```bash
omnicoach-sft-inspect /path/to/OmniCoach-240K/sft/parquet/seed/part_01 --validate
```

To train, use the downloaded part directories as `dataset:` entries in a copy of the [SFT config](../sft/config/train/qwen3_omni_agent_sft.yaml), or pass those directories to `scripts/train.sh` with `--dataset`. See the [SFT README](../sft/README.md#cli-tools--training) for the launcher and the exporter for custom trajectories.

## Download RL tasks and media manifests

```bash
hf download Open-OmniCoach/OmniCoach-240K \
  rl/train.parquet media/media_sources.jsonl media/timelens2_audio_ids.txt \
  --repo-type dataset --local-dir ./OmniCoach-240K
```

Read the released RL table with `pyarrow`:

```python
import pyarrow.parquet as pq

tasks = pq.ParquetFile("./OmniCoach-240K/rl/train.parquet")
batch = next(tasks.iter_batches(batch_size=3, columns=["uid", "question_type", "video_id"]))
print(batch.to_pylist())
```

For rollouts, resolve the source media using `video_source` and `video_id`, then prepare the recipe's task JSONL with the required media references. The [RL README](../rl/README.md) covers setup and launch, and the [Agent-GSPO data validator](../rl/examples/_common/agent_recipe.py) defines the JSONL fields and media references accepted by that recipe.

## Acquire original media

The media manifest covers 22,152 unique videos. Obtain the original videos from their official releases and match on `video_source` and `video_id`. Use `used_in` in `media/media_sources.jsonl` to select the videos for your workflow.

| `video_source` | Unique videos | Official release |
| --- | ---: | --- |
| `timelens2` | 15,419 | [MCG-NJU/TimeLens2-93K](https://huggingface.co/datasets/MCG-NJU/TimeLens2-93K). Use `timelens2_tar_shard` in the manifest to identify the archive. |
| `omnivideo_100k` | 4,546 | [MiG-NJU/OmniVideo-100K](https://huggingface.co/datasets/MiG-NJU/OmniVideo-100K) |
| `cinepile` | 1,308 | [tomg-group-umd/cinepile](https://huggingface.co/datasets/tomg-group-umd/cinepile). Accept the dataset terms on Hugging Face. |
| `cgbench` | 879 | [CG-Bench/CG-Bench](https://huggingface.co/datasets/CG-Bench/CG-Bench). Accept the dataset terms on Hugging Face. |

TimeLens2 distributes video-only files. `media/timelens2_audio_ids.txt` lists the corresponding YouTube IDs, and `audio_requires_separate_download` identifies rows that need separate audio. When synchronized audio is required, obtain the matching tracks in accordance with the source terms and align them by `video_id`.

See the [trace collection guide](../harness/README.md) for the offline harness, the [protocol documentation](protocols.md) for the agent formats, and the [EvalKit](https://github.com/Open-OmniCoach/video-agent-evalkit/tree/omnicoach) for evaluation.
