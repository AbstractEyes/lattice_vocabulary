# ============================================================================
# MANUAL DIFFUSION LOOP - DIRECT UNET CONTROL
# ============================================================================

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import T5EncoderModel, T5Tokenizer
from geovocab2.train.model.relational.cantor_relational import create_cantor_relational
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ----- LOAD PIPELINE COMPONENTS -----
print("Loading SD1.5 components...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
).to('cuda')

# Extract components we need
vae = pipe.vae
unet = pipe.unet
scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer

print("✓ Components loaded")

# ----- LOAD CANTOR MODEL -----
print("\nLoading Cantor model...")
checkpoint = torch.load('./checkpoints/best_model.pt', map_location='cuda')
config = checkpoint['config']

cantor_model = create_cantor_relational(
    dim=config['model_dim'],
    num_heads=config['num_heads'],
    num_blocks=config['num_blocks'],
    seq_len=config['seq_len'],
    cantor_depth=config['cantor_depth'],
    local_window=config['local_window']
)
cantor_model.load_state_dict(checkpoint['model_state_dict'])
cantor_model.to('cuda').eval().half()

print(f"✓ Loaded from step {checkpoint['global_step']}")

# ----- LOAD T5 -----
print("Loading T5...")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
t5_model = T5EncoderModel.from_pretrained("t5-base").to('cuda').eval().half()

print("✓ Setup complete!\n")


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
    text_embeddings = text_encoder(text_input.input_ids.to('cuda'))[0]

    # Negative prompt (empty)
    uncond_input = tokenizer(
        [""],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to('cuda'))[0]

    # Concatenate for classifier-free guidance
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    return text_embeddings


@torch.no_grad()
def get_text_embeddings_cantor(prompt):
    """Get Cantor-modified embeddings."""
    # Get CLIP embeddings
    text_input = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    clip_embed = text_encoder(text_input.input_ids.to('cuda'))[0]

    # Get T5 embeddings
    t5_tokens = t5_tokenizer(
        [prompt],
        max_length=77,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to('cuda')
    t5_embed = t5_model(**t5_tokens).last_hidden_state

    # Process through Cantor
    cantor_embed = cantor_model(clip_embed, t5_embed, return_both=False)

    diff = (cantor_embed - clip_embed).abs()
    print(f"  Cantor change: max={diff.max().item():.4f}, mean={diff.mean().item():.4f}")

    # Negative prompt (empty)
    uncond_input = tokenizer(
        [""],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    clip_embed_uncond = text_encoder(uncond_input.input_ids.to('cuda'))[0]

    t5_tokens_uncond = t5_tokenizer(
        [""],
        max_length=77,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to('cuda')
    t5_embed_uncond = t5_model(**t5_tokens_uncond).last_hidden_state

    cantor_embed_uncond = cantor_model(clip_embed_uncond, t5_embed_uncond, return_both=False)

    # Concatenate for CFG
    text_embeddings = torch.cat([cantor_embed_uncond, cantor_embed])

    return text_embeddings


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

    generator = torch.Generator(device='cuda').manual_seed(seed)

    # Prepare scheduler
    scheduler.set_timesteps(num_steps)

    # Create initial latents
    latents = torch.randn(
        (1, unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        device='cuda',
        dtype=torch.float16
    )

    # Scale initial noise
    latents = latents * scheduler.init_noise_sigma

    # Denoising loop
    for t in tqdm(scheduler.timesteps, desc="Generating"):
        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # Predict noise - THIS IS WHERE WE INJECT OUR EMBEDDINGS
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings  # Our custom embeddings go here!
        ).sample

        # Perform classifier-free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Compute previous noisy sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Decode latents to image
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample

    # Convert to PIL
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")

    from PIL import Image
    image = Image.fromarray(image[0])

    return image


# ----- COMPARISON FUNCTION -----
def generate_comparison(prompt, seed=42):
    """Generate with both methods."""

    print(f"\n{'=' * 60}")
    print(f"PROMPT: {prompt}")
    print(f"{'=' * 60}")

    # Standard CLIP
    print("\n[Standard CLIP]")
    standard_embeds = get_text_embeddings_standard(prompt)
    image_standard = generate_image(prompt, standard_embeds, seed=seed)

    # Cantor
    print("\n[Cantor Relational]")
    cantor_embeds = get_text_embeddings_cantor(prompt)
    image_cantor = generate_image(prompt, cantor_embeds, seed=seed)

    # Display
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image_standard)
    axes[0].set_title("Standard CLIP", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(image_cantor)
    axes[1].set_title("Cantor Relational", fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.suptitle(prompt, fontsize=12, y=0.98)
    plt.tight_layout()
    plt.show()


prompts = [
    "a beautiful sunset over mountains",
    "abstract geometric patterns in vibrant colors",
    "a futuristic cityscape at night",
    "flowing water in a crystal clear stream",
    # "ancient temple ruins covered in vines",
    "A lone cybernetic deer with glimmering silver antlers stands beneath a fractured aurora sky, surrounded by glowing fungal trees, floating quartz shards, and bio-luminescent fog. In the distance, ruined monoliths pulse faint glyphs of a forgotten language, while translucent jellyfish swim through the air above a reflective obsidian lake. The atmosphere is electric with tension, color-shifting through prismatic hues. Distant thunderclouds churn violently."
]  # * 200  # 1000 samples
# ----- TEST -----
# prompts = [
#    "a beautiful sunset over mountains",
#    "a robot playing piano",
#    "a cat wearing sunglasses"
# ]

for prompt in prompts:
    generate_comparison(prompt, seed=42)

print("\n✓ Complete!")