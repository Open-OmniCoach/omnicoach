"""Use Swift's CLI for YAML parsing and torchrun launch."""
import sys

from swift.cli.main import cli_main

from omnicoach_sft.train.config import flat_train_config

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2].endswith((".yaml", ".yml")):
        sys.argv[2] = str(flat_train_config(sys.argv[2]))
    cli_main({"sft": "omnicoach_sft.train.train"}, is_megatron=True)
