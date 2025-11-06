# ============================================================================
# VAE LYRA + SD1.5 INFERENCE - GEOMETRIC EMBEDDING TRANSFORMATION
# ============================================================================

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import T5EncoderModel, T5Tokenizer
from geovocab2.train.trainers.vae.lyra import load_lyra_from_hub
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from PIL import Image
import numpy as np

# ----- CONFIGURATION -----
HF_REPO = "AbstractPhil/vae-lyra"
DEVICE = 'cuda'
DTYPE = torch.float16

# ----- LOAD PIPELINE COMPONENTS -----
print("🎨 Loading SD1.5 components...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=DTYPE,
    safety_checker=None
).to(DEVICE)

# Extract components
vae = pipe.vae
unet = pipe.unet
scheduler = DDIMScheduler.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="scheduler"
)
text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer

print("✓ SD1.5 loaded")

# ----- LOAD VAE LYRA -----
print(f"\n🎵 Loading VAE Lyra from HuggingFace...")
try:
    lyra_model = load_lyra_from_hub(repo_id=HF_REPO, device=DEVICE)
    lyra_model.eval().half()
    print("✓ VAE Lyra loaded from Hub")
except Exception as e:
    print(f"⚠️  Could not load from Hub: {e}")
    print("   Please ensure model exists at:", HF_REPO)
    raise

# ----- LOAD T5 -----
print("\n📝 Loading T5-base...")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
t5_model = T5EncoderModel.from_pretrained("t5-base").to(DEVICE).eval().half()

print("✓ T5 loaded")
print("\n✨ Setup complete!\n")


# ----- ENCODING FUNCTIONS -----
@torch.no_grad()
def get_text_embeddings_standard(prompt):
    """Get standard CLIP embeddings."""
    # Positive prompt
    text_input = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    text_embeddings = text_encoder(text_input.input_ids.to(DEVICE))[0]

    # Negative prompt (empty)
    uncond_input = tokenizer(
        [""],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(DEVICE))[0]

    # Concatenate for classifier-free guidance
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    return text_embeddings


@torch.no_grad()
def get_text_embeddings_lyra(prompt):
    """Get VAE Lyra transformed embeddings."""
    # Get CLIP embeddings
    text_input = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    clip_embed = text_encoder(text_input.input_ids.to(DEVICE))[0]

    # Get T5 embeddings
    t5_tokens = t5_tokenizer(
        [prompt],
        max_length=77,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(DEVICE)
    t5_embed = t5_model(**t5_tokens).last_hidden_state

    # Process through VAE Lyra
    modality_inputs = {
        'clip': clip_embed,
        't5': t5_embed
    }

    # Encode to latent space and decode back to CLIP
    reconstructions, mu, logvar = lyra_model(modality_inputs, target_modalities=['clip'])
    lyra_embed = reconstructions['clip']

    # Compute statistics
    diff = (lyra_embed - clip_embed).abs()
    cos_sim = torch.nn.functional.cosine_similarity(
        lyra_embed.flatten(1),
        clip_embed.flatten(1),
        dim=-1
    ).mean()

    print(f"  Lyra transform: max_Δ={diff.max().item():.4f}, "
          f"mean_Δ={diff.mean().item():.4f}, cos_sim={cos_sim.item():.4f}")

    # Process empty prompt (unconditional)
    uncond_input = tokenizer(
        [""],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    clip_embed_uncond = text_encoder(uncond_input.input_ids.to(DEVICE))[0]

    t5_tokens_uncond = t5_tokenizer(
        [""],
        max_length=77,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(DEVICE)
    t5_embed_uncond = t5_model(**t5_tokens_uncond).last_hidden_state

    modality_inputs_uncond = {
        'clip': clip_embed_uncond,
        't5': t5_embed_uncond
    }

    reconstructions_uncond, _, _ = lyra_model(
        modality_inputs_uncond,
        target_modalities=['clip']
    )
    lyra_embed_uncond = reconstructions_uncond['clip']

    # Concatenate for CFG
    text_embeddings = torch.cat([lyra_embed_uncond, lyra_embed])

    return text_embeddings


@torch.no_grad()
def get_text_embeddings_lyra_latent(prompt):
    """
    Get VAE Lyra latent space directly (for exploration).
    Returns mu (mean of latent distribution).
    """
    text_input = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    clip_embed = text_encoder(text_input.input_ids.to(DEVICE))[0]

    t5_tokens = t5_tokenizer(
        [prompt],
        max_length=77,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(DEVICE)
    t5_embed = t5_model(**t5_tokens).last_hidden_state

    modality_inputs = {
        'clip': clip_embed,
        't5': t5_embed
    }

    mu, logvar = lyra_model.encode(modality_inputs)

    return mu, logvar


# ----- MANUAL DIFFUSION LOOP -----
@torch.no_grad()
def generate_image(
        prompt,
        text_embeddings,
        height=512,
        width=512,
        num_steps=30,
        guidance_scale=7.5,
        seed=42
):
    """Manual diffusion loop with direct UNet control."""

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # Prepare scheduler
    scheduler.set_timesteps(num_steps)

    # Create initial latents
    latents = torch.randn(
        (1, unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        device=DEVICE,
        dtype=DTYPE
    )

    # Scale initial noise
    latents = latents * scheduler.init_noise_sigma

    # Denoising loop
    for t in tqdm(scheduler.timesteps, desc="Generating", leave=False):
        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # Predict noise - INJECT OUR TRANSFORMED EMBEDDINGS
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings
        ).sample

        # Perform classifier-free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
        )

        # Compute previous noisy sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Decode latents to image
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample

    # Convert to PIL
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")

    image = Image.fromarray(image[0])

    return image


# ----- COMPARISON FUNCTION -----
def generate_comparison(prompt, seed=42, num_steps=30, guidance_scale=7.5):
    """Generate with both standard CLIP and VAE Lyra."""

    print(f"\n{'=' * 70}")
    print(f"PROMPT: {prompt}")
    print(f"{'=' * 70}")

    # Standard CLIP
    print("\n[1/2] Standard CLIP")
    standard_embeds = get_text_embeddings_standard(prompt)
    image_standard = generate_image(
        prompt, standard_embeds,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        seed=seed
    )

    # VAE Lyra
    print("\n[2/2] VAE Lyra (Geometric Fusion)")
    lyra_embeds = get_text_embeddings_lyra(prompt)
    image_lyra = generate_image(
        prompt, lyra_embeds,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        seed=seed
    )

    # Display side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(image_standard)
    axes[0].set_title("Standard CLIP", fontsize=16, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(image_lyra)
    axes[1].set_title("VAE Lyra 🎵", fontsize=16, fontweight='bold')
    axes[1].axis('off')

    plt.suptitle(prompt, fontsize=13, y=0.98)
    plt.tight_layout()
    plt.show()

    return image_standard, image_lyra


# ----- BATCH GENERATION -----
def generate_batch_comparison(
        prompts,
        seed=42,
        num_steps=30,
        guidance_scale=7.5,
        save_path=None
):
    """Generate comparisons for multiple prompts."""

    results = []

    for i, prompt in enumerate(prompts):
        print(f"\n{'=' * 70}")
        print(f"[{i + 1}/{len(prompts)}] {prompt}")
        print(f"{'=' * 70}")

        img_standard, img_lyra = generate_comparison(
            prompt,
            seed=seed,
            num_steps=num_steps,
            guidance_scale=guidance_scale
        )

        results.append({
            'prompt': prompt,
            'standard': img_standard,
            'lyra': img_lyra
        })

        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(exist_ok=True)

            img_standard.save(save_dir / f"{i:03d}_standard.png")
            img_lyra.save(save_dir / f"{i:03d}_lyra.png")

    return results


# ----- LATENT SPACE EXPLORATION -----
def explore_latent_space(prompts):
    """Explore VAE Lyra's latent space for multiple prompts."""

    print("\n🔍 Exploring VAE Lyra latent space...")

    latents = []

    for prompt in prompts:
        mu, logvar = get_text_embeddings_lyra_latent(prompt)
        latents.append(mu.cpu())

        # Show statistics
        print(f"\n  '{prompt[:50]}...'")
        print(f"    mu: [{mu.min().item():.3f}, {mu.max().item():.3f}], "
              f"mean={mu.mean().item():.3f}, std={mu.std().item():.3f}")
        print(f"    logvar: [{logvar.min().item():.3f}, {logvar.max().item():.3f}], "
              f"mean={logvar.mean().item():.3f}")

    # Compute pairwise similarities
    print("\n📊 Latent space similarity matrix:")
    similarities = torch.zeros(len(prompts), len(prompts))

    for i in range(len(prompts)):
        for j in range(len(prompts)):
            sim = torch.nn.functional.cosine_similarity(
                latents[i].flatten(),
                latents[j].flatten(),
                dim=0
            )
            similarities[i, j] = sim

    print(similarities.numpy())

    return latents, similarities


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # ----- TEST PROMPTS -----
    test_prompts = [
        "a beautiful sunset over mountains",
        "abstract geometric patterns in vibrant colors",
        "a futuristic cityscape at night",
        "flowing water in a crystal clear stream",
        "A lone cybernetic deer with glimmering silver antlers stands beneath a fractured aurora sky, surrounded by glowing fungal trees, floating quartz shards, and bio-luminescent fog. In the distance, ruined monoliths pulse faint glyphs of a forgotten language, while translucent jellyfish swim through the air above a reflective obsidian lake. The atmosphere is electric with tension, color-shifting through prismatic hues. Distant thunderclouds churn violently."
    ]

    # ----- SINGLE COMPARISON -----
    print("\n🎨 SINGLE COMPARISON TEST")
    generate_comparison(
        test_prompts[0],
        seed=42,
        num_steps=30,
        guidance_scale=7.5
    )

    # ----- BATCH COMPARISON -----
    print("\n\n🎨 BATCH COMPARISON TEST")
    results = generate_batch_comparison(
        test_prompts,
        seed=42,
        num_steps=30,
        guidance_scale=7.5,
        save_path="./lyra_results"
    )

    # ----- LATENT SPACE EXPLORATION -----
    print("\n\n🔍 LATENT SPACE EXPLORATION")
    latents, similarities = explore_latent_space(test_prompts[:3])

    print("\n✨ Complete!")