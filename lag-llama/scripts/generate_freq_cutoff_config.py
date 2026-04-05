#!/usr/bin/env python3
"""
Generate a single config file for running frequency-cutoff experiments on the
best-performing model configuration found in the optim-grid search.

Usage
-----
python scripts/generate_freq_cutoff_config.py \
    --n_layer 5 --n_head 5 \
    --base_config configs/lag_llama.json \
    --output_dir configs/freq_cutoff_configs

The script reads the base config, overrides n_layer and n_head, and writes
one JSON file per (n_layer, n_head) pair to the output directory.
"""

import argparse
import json
from pathlib import Path


def generate_freq_cutoff_config(
    n_layer: int,
    n_head: int,
    base_config_path: str,
    output_dir: str,
) -> str:
    """
    Create a config file for a specific (n_layer, n_head) pair.

    Args:
        n_layer: Number of transformer layers.
        n_head: Number of attention heads.
        base_config_path: Path to the base JSON config.
        output_dir: Directory to write the generated config into.

    Returns:
        Absolute path to the generated config file.
    """
    base_path = Path(base_config_path)
    if not base_path.is_absolute():
        base_path = Path.cwd() / base_path

    with open(base_path) as f:
        config = json.load(f)

    config["n_layer"] = n_layer
    config["n_head"] = n_head

    out_path = Path(output_dir)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.mkdir(parents=True, exist_ok=True)

    filename = f"freq_cutoff_l{n_layer}_h{n_head}.json"
    config_file = out_path / filename

    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Config written to: {config_file}")
    print(f"  n_layer = {n_layer}")
    print(f"  n_head  = {n_head}")
    print(f"  Full config: {json.dumps(config, indent=2)}")

    return str(config_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a config file for frequency-cutoff experiments"
    )
    parser.add_argument(
        "--n_layer",
        type=int,
        required=True,
        help="Number of transformer layers for the target model",
    )
    parser.add_argument(
        "--n_head",
        type=int,
        required=True,
        help="Number of attention heads for the target model",
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/lag_llama.json",
        help="Path to the base configuration file (default: configs/lag_llama.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="configs/freq_cutoff_configs",
        help="Directory to save the generated config (default: configs/freq_cutoff_configs)",
    )

    args = parser.parse_args()

    generate_freq_cutoff_config(
        n_layer=args.n_layer,
        n_head=args.n_head,
        base_config_path=args.base_config,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
