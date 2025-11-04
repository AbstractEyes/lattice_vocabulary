#!/usr/bin/env python3
"""
Proper PT to ComfyUI converter - uses diffusers official script
"""
import sys
import torch
import subprocess
from pathlib import Path
from diffusers import StableDiffusionPipeline

# ========================================================================
# CONFIG
# ========================================================================
PT_PATH = r"E:/sd15/sd15_flowmatch_david_weighted_2_e34.pt"
OUTPUT_PATH = r"I:\AIImageGen\AUTOMATIC1111\stable-diffusion-webui\models\Stable-diffusion\sd1\sd15_flowmatch_david_weighted_2_e34_COMFYUI.safetensors"
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
USE_FP16 = True
# ========================================================================

print("=" * 80)
print("PT → COMFYUI CONVERTER (Using diffusers official tools)")
print("=" * 80)

dtype = torch.float16 if USE_FP16 else torch.float32

# Step 1: Load student weights
print("\n[1/3] Loading student checkpoint...")
ckpt = torch.load(PT_PATH, map_location='cpu', weights_only=False)
student_state = ckpt['student']

# Clean keys
cleaned = {}
for k, v in student_state.items():
    k = k.replace('module.', '').replace('_orig_mod.', '').replace('unet.', '')
    cleaned[k] = v

if USE_FP16:
    cleaned = {k: v.half() if v.dtype == torch.float32 else v for k, v in cleaned.items()}

print(f"  Loaded {len(cleaned)} keys")

# Step 2: Create diffusers pipeline with student weights
print("\n[2/3] Building diffusers pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype,
    safety_checker=None
)

# Load student into UNet
result = pipe.unet.load_state_dict(cleaned, strict=False)
print(f"  UNet loaded - Missing: {len(result.missing_keys)}, Unexpected: {len(result.unexpected_keys)}")

# Save to temporary directory
temp_dir = Path(r"I:\AIImageGen\AUTOMATIC1111\stable-diffusion-webui\models\Stable-diffusion\sd1\temp")
temp_path = temp_dir / "diffusers_pipe"
temp_path.mkdir(parents=True, exist_ok=True)

print(f"  Saving to temp: {temp_path}")
pipe.save_pretrained(temp_path, safe_serialization=True)

# Step 3: Load original SD1.5 as base
print(f"\n[3/4] Loading original SD1.5 as base...")
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

sd15_path = hf_hub_download(
    "runwayml/stable-diffusion-v1-5",
    "v1-5-pruned-emaonly.safetensors",
    repo_type="model"
)

original_sd = load_file(sd15_path)
print(f"  Loaded {len(original_sd)} keys from original SD1.5")

# Step 4: Manual key conversion (diffusers → CompVis)
print(f"\n[4/4] Converting UNet keys to CompVis format...")


def convert_unet_diffusers_to_compvis(diffusers_state):
    """
    Convert diffusers UNet state dict to CompVis/LDM format
    Based on actual key structure analysis
    """
    compvis = {}

    for key, value in diffusers_state.items():
        new_key = None

        # conv_in → input_blocks.0.0
        if key.startswith('conv_in.'):
            param = key.split('conv_in.')[1]
            new_key = f"input_blocks.0.0.{param}"

        # conv_out + conv_norm_out
        elif key.startswith('conv_norm_out.'):
            new_key = key.replace('conv_norm_out.', 'out.0.')
        elif key.startswith('conv_out.'):
            new_key = key.replace('conv_out.', 'out.2.')

        # time_embedding
        elif key.startswith('time_embedding.linear_'):
            idx = key.split('linear_')[1].split('.')[0]
            param = key.split('.')[-1]
            new_key = f"time_embed.{int(idx) * 2 - 2}.{param}"

        # down_blocks → input_blocks
        elif key.startswith('down_blocks.'):
            parts = key.split('.')
            block_idx = int(parts[1])

            if 'resnets' in key:
                resnet_idx = int(parts[3])
                input_block_num = 1 + block_idx * 3 + resnet_idx
                remainder = '.'.join(parts[4:])

                if remainder.startswith('norm1.'):
                    new_key = f"input_blocks.{input_block_num}.0.in_layers.0.{remainder.split('norm1.')[1]}"
                elif remainder.startswith('conv1.'):
                    new_key = f"input_blocks.{input_block_num}.0.in_layers.2.{remainder.split('conv1.')[1]}"
                elif remainder.startswith('time_emb_proj.'):
                    new_key = f"input_blocks.{input_block_num}.0.emb_layers.1.{remainder.split('time_emb_proj.')[1]}"
                elif remainder.startswith('norm2.'):
                    new_key = f"input_blocks.{input_block_num}.0.out_layers.0.{remainder.split('norm2.')[1]}"
                elif remainder.startswith('conv2.'):
                    new_key = f"input_blocks.{input_block_num}.0.out_layers.3.{remainder.split('conv2.')[1]}"
                elif remainder.startswith('conv_shortcut.'):
                    new_key = f"input_blocks.{input_block_num}.0.skip_connection.{remainder.split('conv_shortcut.')[1]}"

            elif 'attentions' in key:
                attn_idx = int(parts[3])
                input_block_num = 1 + block_idx * 3 + attn_idx
                remainder = '.'.join(parts[4:])
                new_key = f"input_blocks.{input_block_num}.1.{remainder}"

            elif 'downsamplers.0.conv.' in key:
                input_block_num = 3 + block_idx * 3
                param = key.split('downsamplers.0.conv.')[1]
                new_key = f"input_blocks.{input_block_num}.0.op.{param}"

        # mid_block → middle_block
        elif key.startswith('mid_block.'):
            parts = key.split('.')

            if 'resnets.0' in key:
                remainder = '.'.join(parts[3:])
                if remainder.startswith('norm1.'):
                    new_key = f"middle_block.0.in_layers.0.{remainder.split('norm1.')[1]}"
                elif remainder.startswith('conv1.'):
                    new_key = f"middle_block.0.in_layers.2.{remainder.split('conv1.')[1]}"
                elif remainder.startswith('time_emb_proj.'):
                    new_key = f"middle_block.0.emb_layers.1.{remainder.split('time_emb_proj.')[1]}"
                elif remainder.startswith('norm2.'):
                    new_key = f"middle_block.0.out_layers.0.{remainder.split('norm2.')[1]}"
                elif remainder.startswith('conv2.'):
                    new_key = f"middle_block.0.out_layers.3.{remainder.split('conv2.')[1]}"

            elif 'resnets.1' in key:
                remainder = '.'.join(parts[3:])
                if remainder.startswith('norm1.'):
                    new_key = f"middle_block.2.in_layers.0.{remainder.split('norm1.')[1]}"
                elif remainder.startswith('conv1.'):
                    new_key = f"middle_block.2.in_layers.2.{remainder.split('conv1.')[1]}"
                elif remainder.startswith('time_emb_proj.'):
                    new_key = f"middle_block.2.emb_layers.1.{remainder.split('time_emb_proj.')[1]}"
                elif remainder.startswith('norm2.'):
                    new_key = f"middle_block.2.out_layers.0.{remainder.split('norm2.')[1]}"
                elif remainder.startswith('conv2.'):
                    new_key = f"middle_block.2.out_layers.3.{remainder.split('conv2.')[1]}"

            elif 'attentions.0' in key:
                remainder = '.'.join(parts[3:])
                new_key = f"middle_block.1.{remainder}"

        # up_blocks → output_blocks
        elif key.startswith('up_blocks.'):
            parts = key.split('.')
            block_idx = int(parts[1])

            if 'resnets' in key:
                resnet_idx = int(parts[3])
                output_block_num = block_idx * 3 + resnet_idx
                remainder = '.'.join(parts[4:])

                if remainder.startswith('norm1.'):
                    new_key = f"output_blocks.{output_block_num}.0.in_layers.0.{remainder.split('norm1.')[1]}"
                elif remainder.startswith('conv1.'):
                    new_key = f"output_blocks.{output_block_num}.0.in_layers.2.{remainder.split('conv1.')[1]}"
                elif remainder.startswith('time_emb_proj.'):
                    new_key = f"output_blocks.{output_block_num}.0.emb_layers.1.{remainder.split('time_emb_proj.')[1]}"
                elif remainder.startswith('norm2.'):
                    new_key = f"output_blocks.{output_block_num}.0.out_layers.0.{remainder.split('norm2.')[1]}"
                elif remainder.startswith('conv2.'):
                    new_key = f"output_blocks.{output_block_num}.0.out_layers.3.{remainder.split('conv2.')[1]}"
                elif remainder.startswith('conv_shortcut.'):
                    new_key = f"output_blocks.{output_block_num}.0.skip_connection.{remainder.split('conv_shortcut.')[1]}"

            elif 'attentions' in key:
                attn_idx = int(parts[3])
                output_block_num = block_idx * 3 + attn_idx
                remainder = '.'.join(parts[4:])
                new_key = f"output_blocks.{output_block_num}.1.{remainder}"

            elif 'upsamplers.0.conv.' in key:
                output_block_num = block_idx * 3 + 2
                param = key.split('upsamplers.0.conv.')[1]
                # Upsamplers have different positions
                if block_idx == 0:
                    new_key = f"output_blocks.{output_block_num}.1.conv.{param}"
                elif block_idx == 1:
                    new_key = f"output_blocks.{output_block_num}.2.conv.{param}"
                elif block_idx == 2:
                    new_key = f"output_blocks.{output_block_num}.2.conv.{param}"

        if new_key:
            compvis[new_key] = value
        else:
            print(f"  Warning: Unmapped key: {key}")

    return compvis


# Get trained UNet state
trained_unet_state = pipe.unet.state_dict()

# Convert to CompVis format
compvis_unet = convert_unet_diffusers_to_compvis(trained_unet_state)
print(f"  Converted {len(compvis_unet)} UNet keys")

# Build final checkpoint
new_checkpoint = {}

# Copy non-UNet keys from original SD
for key, value in original_sd.items():
    if not key.startswith('model.diffusion_model.'):
        new_checkpoint[key] = value

# Add converted UNet with prefix
for key, value in compvis_unet.items():
    new_checkpoint[f"model.diffusion_model.{key}"] = value

print(f"  Final checkpoint: {len(new_checkpoint)} keys")

# Save
print(f"\nSaving to: {OUTPUT_PATH}")
Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
save_file(new_checkpoint, str(OUTPUT_PATH))

# Validation - check key structure matches original SD
print(f"\n" + "=" * 80)
print("VALIDATION: Checking key structure")
print("=" * 80)

original_keys = set(original_sd.keys())
converted_keys = set(new_checkpoint.keys())

# Get UNet keys only
original_unet_keys = {k for k in original_keys if k.startswith('model.diffusion_model.')}
converted_unet_keys = {k for k in converted_keys if k.startswith('model.diffusion_model.')}

missing_keys = original_unet_keys - converted_unet_keys
extra_keys = converted_unet_keys - original_unet_keys

print(f"\nOriginal SD UNet keys: {len(original_unet_keys)}")
print(f"Converted UNet keys: {len(converted_unet_keys)}")
print(f"Missing keys: {len(missing_keys)}")
print(f"Extra keys: {len(extra_keys)}")

if missing_keys or extra_keys:
    print(f"\n❌ VALIDATION FAILED!")

    if missing_keys:
        print(f"\n⚠️  MISSING KEYS ({len(missing_keys)} total):")
        for i, key in enumerate(sorted(missing_keys)[:20], 1):
            print(f"  {i}. {key}")
        if len(missing_keys) > 20:
            print(f"  ... and {len(missing_keys) - 20} more")

    if extra_keys:
        print(f"\n⚠️  EXTRA KEYS ({len(extra_keys)} total):")
        for i, key in enumerate(sorted(extra_keys)[:20], 1):
            print(f"  {i}. {key}")
        if len(extra_keys) > 20:
            print(f"  ... and {len(extra_keys) - 20} more")

    print(f"\n" + "=" * 80)
    print("Conversion produced mismatched keys! Deleting output and exiting.")
    print("=" * 80)

    # Delete the invalid output
    if Path(OUTPUT_PATH).exists():
        Path(OUTPUT_PATH).unlink()
        print(f"Deleted invalid checkpoint: {OUTPUT_PATH}")

    exit(1)

print(f"\n✓ VALIDATION PASSED - All UNet keys match!")

# Optional: Clean up temp directory
# import shutil
# shutil.rmtree(temp_path)

size_gb = Path(OUTPUT_PATH).stat().st_size / 1024 ** 3
print(f"\n{'=' * 80}")
print(f"✓ CONVERSION COMPLETE!")
print(f"{'=' * 80}")
print(f"Output: {OUTPUT_PATH}")
print(f"Size: {size_gb:.2f} GB")
print(f"\nReady for ComfyUI!")