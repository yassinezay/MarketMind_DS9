import requests

def generate_visual_prompt_locally(product_description: str, motif: str):
    structured_prompt = f"""
You are a world-class creative director and AI prompt engineer.

🎯 Your goal: Create a **short, powerful visual prompt** (≤ 60 words) that transforms the following product and emotion into a **cinematic, high-end commercial image**.

🔍 Focus on:
- Photorealism, dramatic lighting, detailed setting
- Brand visibility, accurate logo placement
- Emotive storytelling through visuals
- Composition: centered subject, depth of field, lens flare, or contrast

🛍️ **Product Description**:
{product_description}

🎨 **Emotion / Style to Evoke**:
{motif}

✍️ Write only the visual prompt for Stable Diffusion or Midjourney. No preambles, no labels. Start directly.
"""
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.1:8b", "prompt": structured_prompt, "stream": False}
    )

    return response.json().get("response", "").strip()
