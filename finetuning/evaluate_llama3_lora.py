# --- Step 0: Install Missing Packages
#!pip install evaluate --quiet
#!pip install transformers datasets peft bitsandbytes accelerate --quiet

# --- Step 1: Import Libraries
import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- Step 2: Set HuggingFace Token
os.environ["HF_TOKEN"] = "Api-Token-Here"  # Replace with your Hugging Face token

# --- Step 3: Define Model Paths
finetuned_model_path = "/content/drive/MyDrive/finetuned-llama3-products"
model_id = "meta-llama/Llama-3.1-8B-Instruct"

# --- Step 4: Load Base Model
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=os.environ.get("HF_TOKEN"),
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=os.environ.get("HF_TOKEN"),
    trust_remote_code=True,
    use_fast=True
)
tokenizer.pad_token = tokenizer.eos_token

# --- Step 5: Load Fine-tuned LoRA
#!mkdir -p /content/offload

model = PeftModel.from_pretrained(
    base_model,
    finetuned_model_path,
    device_map="auto",
    offload_folder="/content/offload",
    offload_state_dict=True,
)

# --- Step 6: Define Generation Function
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.5,   # Lower temperature for better reliability
        top_p=0.9,
        repetition_penalty=1.15
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# --- Step 7: Custom Prompt Template
CUSTOM_PROMPT_TEMPLATE = """
🎯 You are a top-tier creative copywriter, specializing in writing engaging and persuasive product ads.

Your mission is to **generate a captivating product ad** using the given information below.
Be creative, highlight unique selling points, and make it emotionally appealing to the audience. 🌟

---

🏢 Company Context:
{context}

🛍️ Product Description:
{question}

🎨 Motif (Tone/Emotion to Use):
{motif}

---

✨ Rules:
- Start with a catchy headline
- Highlight key features naturally
- Make it feel persuasive but NOT robotic
- Use the motif/emotion strongly
- Keep it concise (max 120 words)
- Sprinkle emojis naturally
- Avoid any unrelated content

---

Now, generate the product ad below:
"""

# --- Step 8: Load Test Dataset
csv_path = "/content/drive/MyDrive/products_100_combined.csv"
df = pd.read_csv(csv_path)
df = df.dropna(subset=['description', 'ad'])
df["motif"] = "Excitement"

# --- Step 9: Build Prompts
def build_prompt(row):
    return CUSTOM_PROMPT_TEMPLATE.format(
        context="A modern company offering premium quality products.",
        question=row['description'],
        motif=row['motif']
    )

df["prompt"] = df.apply(build_prompt, axis=1)

# --- Step 10: Test Single Example
sample_prompt = df["prompt"].iloc[0]

print("\n🚀 Generating Ad...")
generated_output = generate_text(sample_prompt)

# --- Step 11: Clean the Output
if "Now, generate the product ad below:" in generated_output:
    generated_ad = generated_output.split("Now, generate the product ad below:")[-1].strip()
else:
    generated_ad = generated_output.strip()

# --- Step 12: Display
print("\n📝 Generated Ad:\n")
print(generated_ad)

print("\n📋 Real Ad (Ground Truth):\n")
print(df["ad"].iloc[0])
