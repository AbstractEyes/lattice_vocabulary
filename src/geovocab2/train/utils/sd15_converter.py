#!/usr/bin/env python3
"""
Convert .pt checkpoint to proper SD1.5 format and upload to HuggingFace
WITH PROPER WEIGHT LOADING VERIFICATION
"""

import sys
import os
import torch
import urllib.request
import subprocess
import shutil
from pathlib import Path
from safetensors.torch import save_file, load_file
from huggingface_hub import HfApi
from google.colab import userdata
from diffusers import UNet2DConditionModel, StableDiffusionPipeline

# CONFIGURE YOUR REPO HERE
REPO_ID = "AbstractPhil/sd15-flow-matching"


def download_converter():
    """Download official conversion script."""
    url = "https://raw.githubusercontent.com/huggingface/diffusers/main/scripts/convert_diffusers_to_original_stable_diffusion.py"
    output = "convert_diffusers_to_original_stable_diffusion.py"

    if Path(output).exists():
        return output

    print("📥 Downloading converter script...")
    urllib.request.urlretrieve(url, output)
    print("✓ Downloaded converter")
    return output


def create_pipeline_from_checkpoint(checkpoint_path: str, output_dir: str):
    """Create diffusers pipeline from student checkpoint."""

    print(f"\n📦 Loading checkpoint: {Path(checkpoint_path).name}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract student state dict
    if 'student' in checkpoint:
        student_state = checkpoint['student']
    elif 'model_state_dict' in checkpoint:
        student_state = checkpoint['model_state_dict']
    else:
        student_state = checkpoint

    print(f"✓ Loaded {len(student_state)} keys from checkpoint")

    # Show sample keys to diagnose format
    sample_keys = list(student_state.keys())[:5]
    print(f"\n📋 Sample student keys:")
    for key in sample_keys:
        print(f"   {key}")

    # Load base UNet
    print(f"\n📥 Loading base SD1.5 UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="unet",
        torch_dtype=torch.float16
    )

    # Get a sample weight BEFORE loading to compare
    sample_param_name = "conv_in.weight"
    if hasattr(unet, 'conv_in') and hasattr(unet.conv_in, 'weight'):
        before_weight = unet.conv_in.weight.clone()
    else:
        before_weight = None

    # Load student weights into UNet
    print(f"\n🔄 Loading student weights into UNet...")
    missing_keys, unexpected_keys = unet.load_state_dict(student_state, strict=False)

    print(f"   Missing keys: {len(missing_keys)}")
    print(f"   Unexpected keys: {len(unexpected_keys)}")

    if len(missing_keys) > 0:
        print(f"\n   Sample missing keys:")
        for key in list(missing_keys)[:10]:
            print(f"      {key}")

    if len(unexpected_keys) > 0:
        print(f"\n   Sample unexpected keys:")
        for key in list(unexpected_keys)[:10]:
            print(f"      {key}")

    # Verify weights actually changed
    if before_weight is not None:
        after_weight = unet.conv_in.weight
        weight_diff = torch.abs(before_weight - after_weight).max().item()
        print(f"\n🔍 Weight change verification:")
        print(f"   conv_in.weight max diff: {weight_diff:.6e}")

        if weight_diff < 1e-8:
            print(f"\n❌ CRITICAL ERROR: Weights did NOT change!")
            print(f"   Student weights were NOT loaded into UNet!")
            print(f"   This means the key format doesn't match.")
            raise ValueError("Student weights failed to load - key format mismatch")

    print(f"✓ Student weights loaded successfully")

    # Load full pipeline
    print(f"\n📥 Loading base SD1.5 pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    )

    # Replace UNet with student
    pipe.unet = unet
    print(f"✓ Replaced UNet with student")

    # Save pipeline
    print(f"\n💾 Saving pipeline to: {output_dir}")
    pipe.save_pretrained(output_dir, safe_serialization=True)
    print(f"✓ Pipeline saved")

    return output_dir


def convert_pipeline_to_checkpoint(converter_script: str, pipeline_path: str, output_path: str):
    """Convert pipeline to original SD checkpoint format."""

    print(f"\n🔄 Converting to checkpoint format...")

    # Force .ckpt output first, then we'll convert
    temp_ckpt = "./temp_output.ckpt"

    # Run converter - it outputs .ckpt format
    cmd = [
        sys.executable,
        converter_script,
        "--model_path", str(Path(pipeline_path).absolute()),
        "--checkpoint_path", temp_ckpt,
        "--half"
    ]

    print(f"   Running converter...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Converter failed: {result.stderr}")
        return False

    print(f"✓ Converter finished")

    # Find the actual output file
    if Path(temp_ckpt).exists():
        checkpoint_file = temp_ckpt
    elif Path(temp_ckpt.replace('.ckpt', '.safetensors')).exists():
        checkpoint_file = temp_ckpt.replace('.ckpt', '.safetensors')
    elif Path(temp_ckpt.replace('.ckpt', '')).exists():
        checkpoint_file = temp_ckpt.replace('.ckpt', '')
    else:
        print(f"❌ Could not find output file")
        return False

    print(f"   Found: {checkpoint_file}")

    # Now convert to proper safetensors
    print(f"🔄 Converting to safetensors...")

    try:
        # Try to load as safetensors first
        state_dict = load_file(checkpoint_file)
        print(f"   ✓ Already in safetensors format")
    except:
        # It's a .ckpt file, load with torch
        print(f"   Loading as PyTorch checkpoint...")
        checkpoint = torch.load(checkpoint_file, map_location='cpu')

        # Extract state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"   ✓ Extracted state_dict")
        else:
            state_dict = checkpoint
            print(f"   ✓ Using full checkpoint as state_dict")

    # Save as proper safetensors
    print(f"   Saving as safetensors: {output_path}")
    save_file(state_dict, output_path)

    # Verify
    try:
        test = load_file(output_path)
        size_gb = Path(output_path).stat().st_size / 1024 ** 3
        print(f"✓ Verified: {len(test)} keys, {size_gb:.2f} GB")

        # Clean up temp file
        if Path(checkpoint_file).exists():
            Path(checkpoint_file).unlink()

        return True
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def upload_checkpoint(checkpoint_path: str):
    """Main function: convert and upload."""

    # Get HF token
    print("🔑 Getting HuggingFace token...")
    try:
        hf_token = userdata.get('HF_TOKEN')
        os.environ['HF_TOKEN'] = hf_token
        print("✓ Token loaded")
    except Exception as e:
        print(f"❌ Failed to get HF_TOKEN: {e}")
        sys.exit(1)

    # Get checkpoint info
    ckpt_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    gstep = ckpt.get('gstep', 0)
    epoch = ckpt.get('epoch', 'unknown')

    print(f"\n📊 Checkpoint info:")
    print(f"   File: {ckpt_path.name}")
    print(f"   Epoch: {epoch}, Step: {gstep}")

    # Temp directories
    temp_pipeline = "./temp_pipeline"
    temp_checkpoint = "./temp_checkpoint.safetensors"

    try:
        # Step 1: Download converter
        converter = download_converter()

        # Step 2: Create pipeline (with verification!)
        create_pipeline_from_checkpoint(checkpoint_path, temp_pipeline)

        # Step 3: Convert to checkpoint
        success = convert_pipeline_to_checkpoint(converter, temp_pipeline, temp_checkpoint)

        if not success:
            print("❌ Conversion failed")
            sys.exit(1)

        # Step 4: Upload to HuggingFace
        print(f"\n🚀 Uploading to {REPO_ID}...")

        upload_filename = f"{ckpt_path.stem}_epoch{epoch}_step{gstep}.safetensors"

        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=temp_checkpoint,
            path_in_repo=upload_filename,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message=f"Upload checkpoint: epoch {epoch}, step {gstep}"
        )

        print(f"\n✅ Upload complete!")
        print(f"🔗 https://huggingface.co/{REPO_ID}")
        print(f"📁 File: {upload_filename}")

    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        for path in [temp_pipeline, temp_checkpoint, "convert_diffusers_to_original_stable_diffusion.py",
                     "temp_output.ckpt"]:
            p = Path(path)
            if p.exists():
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
        print("✓ Cleaned up")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python upload_to_hf.py <checkpoint_path>")
        print(f"\nConfigured repo: {REPO_ID}")
        print(f"\nExample:")
        print(f"  python upload_to_hf.py /content/checkpoints_sd15_flow_david_hf/sd15_flowmatch_david_hf_2.pt")
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    upload_checkpoint(checkpoint_path)