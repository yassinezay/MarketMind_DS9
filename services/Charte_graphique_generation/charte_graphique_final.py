import os, math, torch, gc
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
from diffusers import StableDiffusionPipeline
from IPython.display import display, Image as IPyImage
from safetensors.torch import load_file

def load_model(ckpt_path: str):
    """Charge un modèle Stable Diffusion à partir d'un .ckpt local."""
    pipe = StableDiffusionPipeline.from_single_file(
        ckpt_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    pipe.enable_attention_slicing()
    return pipe


# === UTILS ===
def chunk_input(input_list, max_tokens=77):
    chunk_size = max_tokens - 1
    return [input_list[i * chunk_size:(i + 1) * chunk_size] for i in range(math.ceil(len(input_list) / chunk_size))]

def add_3d_to_prompt(prompt):
    return (
        f"{prompt.strip().rstrip('.')}, ultra photorealistic, hyper-detailed, "
        "professional 3D render, cinematic lighting, ray tracing, high gloss & matte mix, "
        "studio backlight, ambient occlusion, volumetric light, shadow play, "
        "reflections, depth of field, octane render, Adobe Dimension style, "
        "premium product shot, metallic material, futuristic elegance, ultra clean background"
    )

def generate_preview(prompt, size=(700, 400), guidance_scale=15):
    w, h = (size[0] // 8) * 8, (size[1] // 8) * 8
    img = pipe(
        prompt=add_3d_to_prompt(prompt),
        width=w,
        height=h,
        guidance_scale=guidance_scale,
        negative_prompt="blurry, watermark, distorted, cartoon, low detail, poor quality"
    ).images[0]
    return img.resize(size)

def generate_preview_with_chunks(prompt, image_paths, size=(700, 400), max_tokens=77):
    chunks = chunk_input(image_paths, max_tokens)
    final_image = None
    for chunk in chunks:
        chunk_prompt = f"{prompt} Context: {', '.join(Path(p).stem for p in chunk)}"
        final_image = generate_preview(chunk_prompt, size=size)
    return final_image

def apply_soft_effect(image):
    return image.filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.DETAIL)

# === BRAND SETUP ===
def extract_logo_colors(path, n_colors=4):
    from sklearn.cluster import KMeans
    import numpy as np
    img = Image.open(path).convert("RGB").resize((100, 100))
    pixels = np.array(img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors).fit(pixels)
    return ["#" + "".join(f"{int(v):02x}" for v in c) for c in kmeans.cluster_centers_]

def load_font(path, size, fallback_size=36):
    fallback_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
    ]
    try:
        return ImageFont.truetype(path, size)
    except:
        for fb in fallback_paths:
            if os.path.exists(fb):
                return ImageFont.truetype(fb, fallback_size)
        raise RuntimeError("No usable font found.")

brand = {
    "name": "WALLYS",
    "logo_path": "/content/wallys/logo.jpg",
    "colors": extract_logo_colors("/content/wallys/logo.jpg"),
    "typo_title": "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "typo_body": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "style": "Tunisian car manufacturing brand"
}

# === MAIN STRUCTURE ===
def create_style_guide(image_paths):
    canvas = Image.new("RGB", (1800, 1400), "#f9f9f9")
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(brand["typo_title"], 52)
    body_font = load_font(brand["typo_body"], 28)
    footer_font = load_font(brand["typo_body"], 24)

    def section_title(txt, x, y):
        draw.text((x, y), txt.upper(), fill=brand["colors"][1], font=title_font)
        draw.line((x, y+50, x+400, y+50), fill=brand["colors"][1], width=2)
        return y + 70

    def draw_circle(draw, center, radius, fill):
        x, y = center
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=fill)

    def paste_image_with_border(img, x, y, border=6, shadow=True):
        if shadow:
            shadow_img = Image.new("RGBA", (img.width + 20, img.height + 20), (0, 0, 0, 0))
            shadow_layer = Image.new("RGBA", shadow_img.size, (0, 0, 0, 50))
            shadow_img.paste(shadow_layer, (10, 10))
            shadow_img.paste(img.convert("RGBA"), (0, 0))
            canvas.paste(shadow_img.convert("RGB"), (x - 10, y - 10))
        else:
            bordered = Image.new("RGB", (img.width + 2*border, img.height + 2*border), "#dcdcdc")
            canvas.paste(bordered, (x - border, y - border))
            canvas.paste(img, (x, y))
        return y + img.height + 40

    def add_section(title, img_prompt, pos_x, pos_y, size):
        y_pos = section_title(title, pos_x, pos_y)
        img = generate_preview(img_prompt, size=size)
        return paste_image_with_border(img, pos_x, y_pos)

   



