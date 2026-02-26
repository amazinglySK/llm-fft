#!/usr/bin/env python3
"""
Generate temporary config files for systematic grid search over n_layer and n_head.

This script loads the base config and creates temporary configs for all combinations
of n_layer (1-8) and n_head (1-9), ensuring all other parameters stay constant.
"""

import json
import os
from pathlib import Path
from itertools import product


def generate_grid_configs(
    base_config_path: str,
    output_dir: str,
    n_layer_range: tuple = (1, 8),
    n_head_range: tuple = (1, 9),
):
    """
    Generate config files for all combinations of n_layer and n_head.
    
    Args:
        base_config_path: Path to the base configuration file
        output_dir: Directory to save temporary config files
        n_layer_range: Tuple of (min, max) for n_layer (inclusive)
        n_head_range: Tuple of (min, max) for n_head (inclusive)
    
    Returns:
        List of generated config file paths
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)
    
    print(f"Loaded base config from: {base_config_path}")
    print(f"Base config parameters: {json.dumps(base_config, indent=2)}")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate all combinations
    n_layer_values = range(n_layer_range[0], n_layer_range[1] + 1)
    n_head_values = range(n_head_range[0], n_head_range[1] + 1)
    
    generated_configs = []
    total_configs = len(list(n_layer_values)) * len(list(n_head_values))
    
    print(f"Generating {total_configs} config files...")
    print(f"n_layer range: {n_layer_range[0]} to {n_layer_range[1]}")
    print(f"n_head range: {n_head_range[0]} to {n_head_range[1]}")
    print()
    
    count = 0
    for n_layer, n_head in product(n_layer_values, n_head_values):
        # Create modified config
        config = base_config.copy()
        config['n_layer'] = n_layer
        config['n_head'] = n_head
        
        # Generate filename
        config_name = f"grid_l{n_layer}_h{n_head}.json"
        config_path = output_path / config_name
        
        # Write config file
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        generated_configs.append(str(config_path))
        count += 1
        
        if count % 10 == 0 or count == total_configs:
            print(f"  Generated {count}/{total_configs} configs...")
    
    print()
    print(f"✅ Successfully generated {len(generated_configs)} config files in: {output_dir}")
    
    return generated_configs


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate grid search config files for Lag-Llama"
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default="../configs/lag_llama.json",
        help="Path to base configuration file (default: ../configs/lag_llama.json)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../configs/grid_configs",
        help="Directory to save generated configs (default: ../configs/grid_configs)"
    )
    parser.add_argument(
        "--min_n_layer",
        type=int,
        default=1,
        help="Minimum n_layer value (default: 1)"
    )
    parser.add_argument(
        "--max_n_layer",
        type=int,
        default=8,
        help="Maximum n_layer value (default: 8)"
    )
    parser.add_argument(
        "--min_n_head",
        type=int,
        default=1,
        help="Minimum n_head value (default: 1)"
    )
    parser.add_argument(
        "--max_n_head",
        type=int,
        default=9,
        help="Maximum n_head value (default: 9)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    base_config_path = (script_dir / args.base_config).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    
    # Generate configs
    generate_grid_configs(
        base_config_path=str(base_config_path),
        output_dir=str(output_dir),
        n_layer_range=(args.min_n_layer, args.max_n_layer),
        n_head_range=(args.min_n_head, args.max_n_head),
    )


if __name__ == "__main__":
    main()
