#!/usr/bin/env python3
"""
All-in-one: Load student -> Create pipeline -> Convert to checkpoint
WITH PROPER FORMAT HANDLING
"""

import subprocess
import sys
from pathlib import Path
import urllib.request
import torch
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from safetensors.torch import load_file, save_file


def download_converter():
    """Download official conversion script."""
    url = "https://raw.githubusercontent.com/huggingface/diffusers/main/scripts/convert_diffusers_to_original_stable_diffusion.py"
    output = "convert_diffusers_to_original_stable_diffusion.py"

    if Path(output).exists():
        print(f"✓ Converter already downloaded")
        return output

    print(f"📥 Downloading converter...")
    urllib.request.urlretrieve(url, output)
    print(f"   ✓ Saved: {output}")
    return output


def create_pipeline(student_path: str, output_dir: str):
    """Create diffusers pipeline with student UNet."""

    print(f"\n📦 Creating pipeline with student UNet...")

    # Load student
    student_full = load_file(student_path)
    student_unet = {}
    for key, value in student_full.items():
        if key.startswith("model.diffusion_model.unet."):
            new_key = key.replace("model.diffusion_model.unet.", "")
            student_unet[new_key] = value

    print(f"   ✓ Extracted {len(student_unet)} UNet keys")

    # Load base UNet and replace weights
    print(f"   Loading base UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="unet",
        torch_dtype=torch.float16
    )
    unet.load_state_dict(student_unet, strict=False)
    print(f"   ✓ Loaded student weights")

    # Load base pipeline
    print(f"   Loading base pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe.unet = unet

    # Save
    print(f"   Saving pipeline...")
    pipe.save_pretrained(output_dir, safe_serialization=True)
    print(f"   ✓ Saved to: {output_dir}")

    return output_dir


def convert_to_checkpoint(converter_script: str, pipeline_path: str, output_path: str):
    """Convert pipeline to checkpoint."""

    print(f"\n🔄 Converting to checkpoint format...")

    # First convert to .ckpt (the script's default)
    temp_ckpt = output_path.replace('.safetensors', '_temp.ckpt')

    cmd = [
        sys.executable,
        converter_script,
        "--model_path", pipeline_path,
        "--checkpoint_path", temp_ckpt,
        "--half"
    ]

    print(f"   Running converter...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"   ❌ Conversion failed:")
        print(result.stderr)
        return False

    print(f"   ✓ Created .ckpt file")

    # Check if it actually created a .ckpt or .safetensors
    actual_output = None
    for ext in ['.ckpt', '.safetensors', '']:
        test_path = temp_ckpt.replace('_temp.ckpt', ext)
        if Path(test_path).exists():
            actual_output = test_path
            break

    if not actual_output:
        print(f"   ❌ Could not find output file")
        return False

    print(f"   Found output: {actual_output}")

    # Convert to safetensors if needed
    if actual_output.endswith('.ckpt'):
        print(f"   Converting .ckpt to .safetensors...")

        # Load .ckpt
        checkpoint = torch.load(actual_output, map_location='cpu')

        # Extract state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Save as safetensors
        save_file(state_dict, output_path)
        print(f"   ✓ Saved as safetensors")

        # Clean up temp .ckpt
        Path(actual_output).unlink()
        print(f"   ✓ Cleaned up temp .ckpt")
    else:
        # Already safetensors, just rename
        import shutil
        shutil.move(actual_output, output_path)
        print(f"   ✓ Moved to final location")

    # Verify the file
    print(f"\n   Verifying safetensors file...")
    try:
        test_load = load_file(output_path)
        print(f"   ✓ File is valid! ({len(test_load)} keys)")

        # Check size
        size_gb = Path(output_path).stat().st_size / 1024 ** 3
        print(f"   ✓ Size: {size_gb:.2f} GB")

        return True
    except Exception as e:
        print(f"   ❌ File verification failed: {e}")
        return False


if __name__ == "__main__":
    # HARDCODED PATHS
    student_checkpoint = r"I:\AIImageGen\AUTOMATIC1111\stable-diffusion-webui\models\Stable-diffusion\sd1\sd15_flowmatch_david_hf_1.safetensors"
    temp_pipeline = r"E:\mirel\crystal_lattice\temp_student_pipeline"
    output_checkpoint = r"I:\AIImageGen\AUTOMATIC1111\stable-diffusion-webui\models\Stable-diffusion\sd1\sd15_flow_WORKING.safetensors"

    print("=" * 80)
    print("STUDENT -> COMFYUI CHECKPOINT CONVERTER (FIXED)")
    print("=" * 80)

    print(f"\nPaths:")
    print(f"  Student:  {student_checkpoint}")
    print(f"  Temp:     {temp_pipeline}")
    print(f"  Output:   {output_checkpoint}")

    # Step 1: Download converter
    converter = download_converter()

    # Step 2: Create pipeline
    pipeline_dir = create_pipeline(student_checkpoint, temp_pipeline)

    # Step 3: Convert to checkpoint
    success = convert_to_checkpoint(converter, pipeline_dir, output_checkpoint)

    if success:
        print(f"\n✅ ALL DONE!")
        print(f"🎯 ComfyUI checkpoint: {output_checkpoint}")
        print(f"   This file should now load in ComfyUI!")

        # Cleanup
        import shutil

        print(f"\n🧹 Cleaning up temp files...")
        if Path(temp_pipeline).exists():
            shutil.rmtree(temp_pipeline)
        if Path(converter).exists():
            Path(converter).unlink()
        print(f"   ✓ Cleaned up")
    else:
        print(f"\n❌ Conversion failed!")
        print(f"   Temp pipeline saved at: {temp_pipeline}")
        print(f"   You can try converting manually")
        sys.exit(1)