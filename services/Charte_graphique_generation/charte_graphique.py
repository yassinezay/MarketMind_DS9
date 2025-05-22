from diffusers import StableDiffusionPipeline
import torch, os
from PIL import Image

# === Load your local model ===
pipe = StableDiffusionPipeline.from_single_file(
    "models/wallyscar_session.ckpt",
    torch_dtype=torch.float16,
    safety_checker=None,  # explicitly skip if needed
    use_safetensors=False  # set to False if it's NOT a .safetensors model
).to("cuda")
pipe.enable_attention_slicing()

# === Generation function using that pipeline ===
def generate_sd_image(prompt, size=(700, 400), guidance_scale=7.5, num_inference_steps=30):
    print(f"[GENERATING with SD] {prompt}")
    image = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images[0]
    image = image.resize(size)
    return image