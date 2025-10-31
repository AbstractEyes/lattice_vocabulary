#!/usr/bin/env python3
"""
Test .pt checkpoint before conversion
Loads UNet from .pt + VAE/CLIP from base SD1.5
Uses manual denoising loop with timestep shift control
"""
import sys
import torch
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, UNet2DConditionModel


def convert_original_sd_to_diffusers(checkpoint, device="cpu"):
    """Convert original SD checkpoint keys to diffusers format."""
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

    print(f"   ⏳ Converting original SD format to diffusers...")
    pipe = download_from_original_stable_diffusion_ckpt(
        checkpoint_path_or_dict=checkpoint,
        device=device,
        from_safetensors=False,
        load_safety_checker=False,
        prediction_type="v_prediction",
    )
    return pipe


def generate_with_shift(pipe, prompt, steps=20, cfg=7.5, shift=3.0, seed=42):
    """Manual generation with timestep shift control."""
    device = pipe.device
    dtype = pipe.unet.dtype

    print(f"      🔧 Steps: {steps}, CFG: {cfg}, Shift: {shift}, Seed: {seed}")

    # Encode prompt
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]

    # Uncond embeddings
    uncond_inputs = pipe.tokenizer(
        "",
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipe.text_encoder(uncond_inputs.input_ids.to(device))[0]

    # Set timesteps
    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps.float().clone()

    # Apply timestep shift
    if shift > 0:
        t_max = timesteps.max()
        t_norm = timesteps / t_max
        # SD3-style shift: t' = shift * t / (1 + (shift - 1) * t)
        t_shifted = shift * t_norm / (1.0 + (shift - 1.0) * t_norm)
        timesteps = (t_shifted * t_max).long()
        print(
            f"      📊 Timesteps (shifted): first={timesteps[0]}, mid={timesteps[len(timesteps) // 2]}, last={timesteps[-1]}")
    else:
        timesteps = timesteps.long()
        print(
            f"      📊 Timesteps (linear):  first={timesteps[0]}, mid={timesteps[len(timesteps) // 2]}, last={timesteps[-1]}")

    # Initial latents
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, 4, 64, 64),
        generator=generator,
        device=device,
        dtype=dtype
    )

    # Scale initial noise
    latents = latents * pipe.scheduler.init_noise_sigma

    # Denoising loop
    print(f"      🎨 Denoising...", end="", flush=True)
    for i, t in enumerate(timesteps):
        # Expand latents for CFG
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict
        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=torch.cat([uncond_embeddings, text_embeddings])
            ).sample

        # CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)

        # Step
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Progress indicator
        if (i + 1) % 5 == 0:
            print(f".", end="", flush=True)

    print(" ✓")

    # Decode
    print(f"      🖼️  Decoding...", end="", flush=True)
    with torch.no_grad():
        image = pipe.vae.decode(latents / 0.18215).sample
    print(" ✓")

    # Convert to PIL
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)


def tst_checkpoint(pt_path, prompt: str = "a beautiful landscape with mountains", timestep_shift: float = 3.0):
    """Load and test a .pt checkpoint with manual denoising."""
    pt_path = Path(pt_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device}")

    # 1. Load checkpoint
    print(f"\n📦 Loading: {pt_path.name}")
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)

    # 2. Check if this is an original SD checkpoint
    has_full_model = any(k.startswith("model.diffusion_model.") or k.startswith("first_stage_model.")
                         for k in ckpt.keys())

    if has_full_model:
        print(f"   ✓ Detected original SD format")
        print(f"\n🏗️ Converting original SD checkpoint to diffusers...")
        pipe = convert_original_sd_to_diffusers(ckpt, device=device)
        print(f"   ✓ Converted and loaded pipeline")
    else:
        # Diffusers format - parse structure
        if "student" in ckpt:
            state_dict = ckpt["student"]
            print(f"   ✓ Found 'student' weights")
            if "gstep" in ckpt:
                print(f"   ✓ Global step: {ckpt['gstep']:,}")
        elif "unet" in ckpt or "state_dict" in ckpt:
            state_dict = ckpt.get("unet") or ckpt.get("state_dict")
            print(f"   ✓ Found direct UNet weights")
        else:
            state_dict = ckpt
            print(f"   ⚠️ Unknown structure, using full checkpoint")

        # Strip any prefixes
        if any(k.startswith("unet.") for k in state_dict.keys()):
            state_dict = {k.replace("unet.", "", 1): v for k, v in state_dict.items()}
            print(f"   ✓ Stripped 'unet.' prefix")

        # Load base pipeline and replace UNet
        print(f"\n🏗️ Loading SD1.5 base...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None
        ).to(device)

        print(f"\n🔄 Loading trained UNet weights...")
        missing, unexpected = pipe.unet.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"   ⚠️ Missing keys: {len(missing)}")
        if unexpected:
            print(f"   ⚠️ Unexpected keys: {len(unexpected)}")
        print(f"   ✓ UNet loaded")

    # Move to device
    pipe = pipe.to(device)

    # 3. Configure Euler scheduler for flow matching
    print(f"\n⚙️ Configuring Euler scheduler for flow matching...")
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
        prediction_type="v_prediction"
    )
    print(f"   ✓ Scheduler: Euler Discrete (v_prediction)")
    print(f"   ✓ Manual denoising with timestep shift control")

    # 4. Generate multiple tests with different seeds
    print(f"\n🎨 Generating test images...")
    print(f"   Prompt: '{prompt}'")
    print(f"   Timestep shift: {timestep_shift}")
    images = []

    for seed in [42, 123, 456]:
        print(f"\n   Seed {seed}:")

        image = generate_with_shift(
            pipe=pipe,
            prompt=prompt,
            steps=20,
            cfg=7.5,
            shift=timestep_shift,
            seed=seed
        )
        images.append(image)

        output = Path(f"test_{pt_path.stem}_seed{seed}_shift{timestep_shift:.1f}.png")
        image.save(output)
        print(f"      ✅ Saved: {output.name}")

    # 5. Create comparison grid
    print(f"\n📊 Creating comparison grid...")
    grid = Image.new('RGB', (1536, 512))
    for i, img in enumerate(images):
        grid.paste(img, (i * 512, 0))

    grid_output = Path(f"test_{pt_path.stem}_grid_shift{timestep_shift:.1f}.png")
    grid.save(grid_output)
    print(f"   ✓ Saved: {grid_output.name}")

    print(f"\n✅ Complete! Generated {len(images)} images + grid")
    return images, grid


if __name__ == "__main__":
    pt_path = r"E:\sd15\sd15_flowmatch_david_weighted_e11.pt"
    prompt = "dimly lit bedroom, girl wearing white dress, sitting on a chair"

    # Test with shift=3.0 (SD3-style)
    print("=" * 70)
    print("Testing with shift=9.0 (SD3-style - bias toward clean samples)")
    print("=" * 70)
    tst_checkpoint(Path(pt_path), prompt, timestep_shift=9.0)

    # Test with shift=3.0 (SD3-style)
    print("=" * 70)
    print("Testing with shift=6.0 (SD3-style - bias toward clean samples)")
    print("=" * 70)
    tst_checkpoint(Path(pt_path), prompt, timestep_shift=6.0)

    # Test with shift=3.0 (SD3-style)
    print("=" * 70)
    print("Testing with shift=3.0 (SD3-style - bias toward clean samples)")
    print("=" * 70)
    tst_checkpoint(Path(pt_path), prompt, timestep_shift=3.0)

    # Test with shift=0.0 (no shift - linear schedule)
    print("\n" + "=" * 70)
    print("Testing with shift=0.0 (linear schedule)")
    print("=" * 70)
    tst_checkpoint(Path(pt_path), prompt, timestep_shift=0.0)

    print("\n🎉 All tests complete!")
    print("Compare the _shift3.0.png vs _shift0.0.png images to see the difference!")