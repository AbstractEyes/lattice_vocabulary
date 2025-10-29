#!/usr/bin/env python3
"""
Block-by-block difference analysis between two checkpoints
Usage: python compare_epochs.py <checkpoint1> <checkpoint2>
"""

import torch
from pathlib import Path
from safetensors.torch import load_file
import numpy as np
from collections import defaultdict


def analyze_checkpoint_differences(ckpt1_path: str, ckpt2_path: str):
    """Compare two checkpoints block-by-block."""

    print("=" * 80)
    print("CHECKPOINT DIFFERENCE ANALYSIS")
    print("=" * 80)

    # Load checkpoints
    print(f"\n📦 Loading checkpoint 1: {Path(ckpt1_path).name}")
    ckpt1 = load_file(ckpt1_path)
    print(f"   ✓ {len(ckpt1)} keys")

    print(f"\n📦 Loading checkpoint 2: {Path(ckpt2_path).name}")
    ckpt2 = load_file(ckpt2_path)
    print(f"   ✓ {len(ckpt2)} keys")

    # Check if keys match
    keys1 = set(ckpt1.keys())
    keys2 = set(ckpt2.keys())

    if keys1 != keys2:
        print(f"\n⚠️  WARNING: Keys don't match!")
        print(f"   Only in ckpt1: {len(keys1 - keys2)}")
        print(f"   Only in ckpt2: {len(keys2 - keys1)}")
        common_keys = keys1 & keys2
    else:
        print(f"\n✓ Keys match perfectly")
        common_keys = keys1

    # Analyze UNet keys only
    unet_keys = [k for k in common_keys if k.startswith("model.diffusion_model.")]
    print(f"\n🔍 Analyzing {len(unet_keys)} UNet keys...")

    # Group by block
    block_stats = defaultdict(lambda: {
        'keys': [],
        'mean_diff': [],
        'max_diff': [],
        'changed': 0,
        'unchanged': 0
    })

    total_params = 0
    total_changed = 0
    identical_keys = []

    for key in unet_keys:
        # Get block name
        parts = key.split('.')
        if 'input_blocks' in key:
            block_idx = parts[parts.index('input_blocks') + 1]
            block_name = f"input_blocks.{block_idx}"
        elif 'middle_block' in key:
            block_name = "middle_block"
        elif 'output_blocks' in key:
            block_idx = parts[parts.index('output_blocks') + 1]
            block_name = f"output_blocks.{block_idx}"
        elif 'time_embed' in key:
            block_name = "time_embed"
        elif 'out.' in key:
            block_name = "output_conv"
        else:
            block_name = "other"

        # Compare tensors
        t1 = ckpt1[key].float()
        t2 = ckpt2[key].float()

        diff = torch.abs(t1 - t2)
        mean_diff = diff.mean().item()
        max_diff = diff.max().item()

        num_params = t1.numel()
        total_params += num_params

        # Check if changed
        if max_diff > 1e-8:  # Changed
            block_stats[block_name]['changed'] += 1
            total_changed += 1
        else:  # Identical
            block_stats[block_name]['unchanged'] += 1
            identical_keys.append(key)

        block_stats[block_name]['keys'].append(key)
        block_stats[block_name]['mean_diff'].append(mean_diff)
        block_stats[block_name]['max_diff'].append(max_diff)

    # Print results
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total UNet keys: {len(unet_keys)}")
    print(f"   Changed keys: {total_changed}")
    print(f"   Unchanged keys: {len(identical_keys)}")
    print(f"   Change rate: {total_changed / len(unet_keys) * 100:.1f}%")

    if len(identical_keys) > 0:
        print(f"\n⚠️  IDENTICAL KEYS (first 20):")
        for key in identical_keys[:20]:
            print(f"      {key}")
        if len(identical_keys) > 20:
            print(f"      ... and {len(identical_keys) - 20} more")

    # Block-by-block analysis
    print(f"\n📊 BLOCK-BY-BLOCK ANALYSIS:")
    print(f"\n{'Block':<20} {'Keys':>6} {'Changed':>8} {'Unchanged':>10} {'Mean Δ':>12} {'Max Δ':>12}")
    print("-" * 80)

    for block_name in sorted(block_stats.keys()):
        stats = block_stats[block_name]
        total_keys = len(stats['keys'])
        changed = stats['changed']
        unchanged = stats['unchanged']

        if len(stats['mean_diff']) > 0:
            avg_mean_diff = np.mean(stats['mean_diff'])
            avg_max_diff = np.mean(stats['max_diff'])
        else:
            avg_mean_diff = 0.0
            avg_max_diff = 0.0

        print(
            f"{block_name:<20} {total_keys:>6} {changed:>8} {unchanged:>10} {avg_mean_diff:>12.6e} {avg_max_diff:>12.6e}")

    # Detailed block analysis
    print(f"\n📋 DETAILED BLOCK ANALYSIS:")

    for block_name in sorted(block_stats.keys()):
        stats = block_stats[block_name]

        print(f"\n  Block: {block_name}")
        print(f"     Total keys: {len(stats['keys'])}")
        print(f"     Changed: {stats['changed']}")
        print(f"     Unchanged: {stats['unchanged']}")

        if stats['changed'] > 0:
            changed_indices = [i for i, d in enumerate(stats['max_diff']) if d > 1e-8]

            # Show top 5 most changed keys
            sorted_indices = sorted(changed_indices, key=lambda i: stats['max_diff'][i], reverse=True)[:5]

            print(f"     Top changed keys:")
            for idx in sorted_indices:
                key = stats['keys'][idx]
                mean_d = stats['mean_diff'][idx]
                max_d = stats['max_diff'][idx]
                short_key = key.replace("model.diffusion_model.", "")
                print(f"        {short_key}")
                print(f"           Mean Δ: {mean_d:.6e}, Max Δ: {max_d:.6e}")

    # Check VAE and text encoder
    print(f"\n📊 OTHER COMPONENTS:")

    vae_keys = [k for k in common_keys if k.startswith("first_stage_model.")]
    text_keys = [k for k in common_keys if k.startswith("cond_stage_model.")]

    vae_changed = 0
    for key in vae_keys:
        diff = torch.abs(ckpt1[key].float() - ckpt2[key].float()).max().item()
        if diff > 1e-8:
            vae_changed += 1

    text_changed = 0
    for key in text_keys:
        diff = torch.abs(ckpt1[key].float() - ckpt2[key].float()).max().item()
        if diff > 1e-8:
            text_changed += 1

    print(f"   VAE: {vae_changed}/{len(vae_keys)} keys changed")
    print(f"   Text Encoder: {text_changed}/{len(text_keys)} keys changed")

    # Diagnosis
    print(f"\n💡 DIAGNOSIS:")
    if total_changed == 0:
        print(f"   ❌ NO CHANGES DETECTED!")
        print(f"      The checkpoints are IDENTICAL")
        print(f"      Training may not be working!")
    elif total_changed < len(unet_keys) * 0.1:
        print(f"   ⚠️  Very few changes detected ({total_changed}/{len(unet_keys)})")
        print(f"      Only {total_changed / len(unet_keys) * 100:.1f}% of keys changed")
        print(f"      Training may be frozen for some blocks")
    else:
        print(f"   ✓ Changes detected in {total_changed / len(unet_keys) * 100:.1f}% of keys")
        print(f"   ✓ Training appears to be working")


if __name__ == "__main__":
    # HARDCODED PATHS - UPDATE THESE
    checkpoint1 = r"I:\AIImageGen\AUTOMATIC1111\stable-diffusion-webui\models\Stable-diffusion\sd1\sd15_flowmatch_david_comfyui_e2.safetensors"
    checkpoint2 = r"I:\AIImageGen\AUTOMATIC1111\stable-diffusion-webui\models\Stable-diffusion\sd1\sd15_flowmatch_david_comfyui_e3.safetensors"

    print("Comparing:")
    print(f"  Checkpoint 1: {checkpoint1}")
    print(f"  Checkpoint 2: {checkpoint2}")
    print()

    if not Path(checkpoint1).exists():
        print(f"❌ Checkpoint 1 not found: {checkpoint1}")
        exit(1)

    if not Path(checkpoint2).exists():
        print(f"❌ Checkpoint 2 not found: {checkpoint2}")
        exit(1)

    analyze_checkpoint_differences(checkpoint1, checkpoint2)