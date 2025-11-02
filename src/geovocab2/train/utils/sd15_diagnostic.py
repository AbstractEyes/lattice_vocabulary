#!/usr/bin/env python3
"""
Diagnostic: Compare your checkpoint keys vs what ComfyUI expects
"""

import torch
from pathlib import Path
from collections import defaultdict

# ========================================================================
# CONFIG
# ========================================================================
PT_PATH = "E:/sd15/sd15_flowmatch_david_weighted_e13.pt"  # Your checkpoint
OUTPUT_CHECKPOINT = "E:/sd15/sd15_flowmatch_david_weighted_e13_FIXED.safetensors"  # The file you tried to load in ComfyUI
# ========================================================================

print("=" * 80)
print("DIAGNOSTIC: Checkpoint Key Analysis")
print("=" * 80)

# 1. Check your training checkpoint
print("\n[1] YOUR TRAINING CHECKPOINT")
print(f"Loading: {PT_PATH}")

try:
    ckpt = torch.load(PT_PATH, map_location='cpu')
    print(f"  Top-level keys: {list(ckpt.keys())}")

    student = ckpt['student']
    print(f"  Student has {len(student)} keys")
    print(f"\n  First 10 student keys:")
    for i, k in enumerate(list(student.keys())[:10]):
        print(f"    {i + 1}. {k}")

except Exception as e:
    print(f"  Error: {e}")

# 2. Check the output checkpoint you created
print("\n" + "=" * 80)
print("[2] YOUR CONVERTED CHECKPOINT")

output_checkpoint_data = None  # Store for later comparison
if Path(OUTPUT_CHECKPOINT).exists():
    print(f"Loading: {OUTPUT_CHECKPOINT}")
    try:
        from safetensors.torch import load_file

        output = load_file(OUTPUT_CHECKPOINT)
        output_checkpoint_data = output  # Save for later
        print(f"  Has {len(output)} keys")
        print(f"\n  First 10 keys:")
        for i, k in enumerate(list(output.keys())[:10]):
            print(f"    {i + 1}. {k}")

        # Check if it has the expected prefixes
        has_unet = any(k.startswith('model.diffusion_model.') for k in output.keys())
        has_vae = any(k.startswith('first_stage_model.') for k in output.keys())
        has_clip = any(k.startswith('cond_stage_model.') for k in output.keys())

        print(f"\n  Structure check:")
        print(f"    Has UNet keys (model.diffusion_model.*): {has_unet}")
        print(f"    Has VAE keys (first_stage_model.*): {has_vae}")
        print(f"    Has CLIP keys (cond_stage_model.*): {has_clip}")

        if not (has_unet and has_vae and has_clip):
            print(f"\n  ⚠ WARNING: Missing expected key prefixes!")

    except Exception as e:
        print(f"  Error: {e}")
else:
    print(f"  File not found: {OUTPUT_CHECKPOINT}")

# 3. Check what a real SD1.5 checkpoint looks like
print("\n" + "=" * 80)
print("[3] REFERENCE: Real SD1.5 Checkpoint Structure")
print("Downloading a real SD1.5 checkpoint to compare...")

try:
    from diffusers import StableDiffusionPipeline
    from huggingface_hub import hf_hub_download

    # Download the actual single-file SD1.5
    print("  Downloading v1-5-pruned-emaonly.safetensors...")
    ckpt_path = hf_hub_download(
        "runwayml/stable-diffusion-v1-5",
        "v1-5-pruned-emaonly.safetensors",
        repo_type="model"
    )

    from safetensors.torch import load_file

    real_sd = load_file(ckpt_path)

    print(f"  Real SD1.5 has {len(real_sd)} keys")
    print(f"\n  First 10 keys:")
    for i, k in enumerate(list(real_sd.keys())[:10]):
        print(f"    {i + 1}. {k}")

    # Analyze structure
    unet_keys = [k for k in real_sd.keys() if 'model.diffusion_model' in k]
    vae_keys = [k for k in real_sd.keys() if 'first_stage_model' in k]
    clip_keys = [k for k in real_sd.keys() if 'cond_stage_model' in k]

    print(f"\n  Key distribution:")
    print(f"    UNet keys: {len(unet_keys)}")
    print(f"    VAE keys: {len(vae_keys)}")
    print(f"    CLIP keys: {len(clip_keys)}")
    print(f"    Other keys: {len(real_sd) - len(unet_keys) - len(vae_keys) - len(clip_keys)}")

    # Now compare with your converted checkpoint in detail
    if output_checkpoint_data is not None:
        print("\n" + "=" * 80)
        print("[4] DETAILED KEY COMPARISON")
        print("=" * 80)

        output = output_checkpoint_data

        real_keys = set(real_sd.keys())
        your_keys = set(output.keys())

        # Keys in real SD but not in yours
        missing_in_yours = real_keys - your_keys
        # Keys in yours but not in real SD
        extra_in_yours = your_keys - real_keys

        print(f"\n📊 Summary:")
        print(f"  Real SD1.5 total keys: {len(real_keys)}")
        print(f"  Your checkpoint total keys: {len(your_keys)}")
        print(f"  Keys in BOTH: {len(real_keys & your_keys)}")
        print(f"  Missing from yours: {len(missing_in_yours)}")
        print(f"  Extra in yours: {len(extra_in_yours)}")

        if missing_in_yours:
            print(f"\n❌ KEYS IN REAL SD1.5 BUT NOT IN YOUR CHECKPOINT ({len(missing_in_yours)} total):")
            for i, k in enumerate(sorted(missing_in_yours), 1):
                print(f"  {i}. {k}")

        if extra_in_yours:
            print(f"\n⚠️  KEYS IN YOUR CHECKPOINT BUT NOT IN REAL SD1.5 ({len(extra_in_yours)} total):")
            for i, k in enumerate(sorted(extra_in_yours), 1):
                print(f"  {i}. {k}")

        # Analyze patterns in missing keys
        if missing_in_yours:
            print(f"\n🔍 ANALYZING MISSING KEY PATTERNS:")

            # Group by prefix
            missing_by_prefix = defaultdict(list)
            for k in missing_in_yours:
                prefix = k.split('.')[0] + '.' + k.split('.')[1] if '.' in k else k
                missing_by_prefix[prefix].append(k)

            for prefix, keys in sorted(missing_by_prefix.items()):
                print(f"\n  Prefix '{prefix}': {len(keys)} keys")
                if len(keys) <= 10:
                    for k in keys:
                        print(f"    - {k}")
                else:
                    for k in keys[:5]:
                        print(f"    - {k}")
                    print(f"    ... and {len(keys) - 5} more")

        # Analyze patterns in extra keys
        if extra_in_yours:
            print(f"\n🔍 ANALYZING EXTRA KEY PATTERNS:")

            extra_by_prefix = defaultdict(list)
            for k in extra_in_yours:
                prefix = k.split('.')[0] + '.' + k.split('.')[1] if '.' in k else k
                extra_by_prefix[prefix].append(k)

            for prefix, keys in sorted(extra_by_prefix.items()):
                print(f"\n  Prefix '{prefix}': {len(keys)} keys")
                if len(keys) <= 10:
                    for k in keys:
                        print(f"    - {k}")
                else:
                    for k in keys[:5]:
                        print(f"    - {k}")
                    print(f"    ... and {len(keys) - 5} more")
    else:
        print("\n⚠️  Skipping comparison - your converted checkpoint was not loaded")

except Exception as e:
    print(f"  Error: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nPlease share this output so we can debug the key structure issue!")