from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import torch

class WallysCarPipeline:
    def __init__(self, model_path: str):
        self.pipeline = StableDiffusionPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.to("cuda")

    def generate(
        self,
        prompt: str,
        output_path: str,
        num_inference_steps: int = 40,
        guidance_scale: float = 8.5,
        seed: int = None,
        negative_prompt: str = "blurry, distorted, low quality, watermark, text, cropped, poorly drawn"
    ):
        generator = torch.Generator("cuda").manual_seed(seed) if seed else None
        with torch.autocast("cuda"):
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )
        result.images[0].save(output_path)
        return output_path

    @staticmethod
    def overlay_logo(base_image_path, logo_path, scale=0.2):
        base_image = Image.open(base_image_path).convert("RGBA")
        logo = Image.open(logo_path).convert("RGBA")

        # Resize logo
        logo = logo.resize((int(base_image.width * scale), int(base_image.height * scale)))

        # Calculate position for top-right corner
        position = (base_image.width - logo.width - 10, 10)  # 10 pixels from the top-right edge

        # Overlay logo on the image
        base_image.paste(logo, position, logo)

        output_path = base_image_path.replace(".png", "_with_logo.png")
        base_image.save(output_path)
        return output_path
