# run this code on google colab using GPU runtime T4
# --- Step 0: Install Required Packages
#%pip install transformers datasets peft bitsandbytes accelerate --quiet

# --- Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# --- Step 2: Import Libraries
import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)
from peft import LoraConfig, get_peft_model

# --- Step 3: Set Hugging Face Token
os.environ["HF_TOKEN"] = "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"  # <-- Secure your key normally!

# --- Step 4: Load and Prepare Dataset
csv_path = "/content/drive/MyDrive/products_100_combined.csv"
df = pd.read_csv(csv_path).dropna(subset=['description', 'ad'])

def format(row):
    return {
        "text": (
            f"### Instruction:\nGenerate a product ad from the description.\n\n"
            f"### Input:\n{row['description']}\n\n"
            f"### Output:\n{row['ad']}"
        )
    }

dataset = Dataset.from_pandas(pd.DataFrame(df.apply(format, axis=1).tolist()))

# --- Step 5: Load Tokenizer
model_id = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=os.environ["HF_TOKEN"],
    trust_remote_code=True,
    use_fast=True
)
tokenizer.pad_token = tokenizer.eos_token

# --- Step 6: Tokenize Dataset
def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize, batched=True, num_proc=2)

# --- Step 7: Load Model (Properly)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=os.environ["HF_TOKEN"],
    trust_remote_code=True,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_8bit=True  # <- important to avoid meta tensor error
    )
)

# --- Step 8: Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# --- Step 9: Define Training Arguments
output_dir = "/content/drive/MyDrive/finetuned-llama3-products"
checkpoint_dir = f"{output_dir}/checkpoints"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=2,                    # 2 epochs for better results
    learning_rate=3e-4,                    # slightly higher for faster convergence
    logging_steps=1,
    save_steps=10,
    save_total_limit=2,
    fp16=False,                            # 8bit quantized models do not need fp16
    bf16=False,
    report_to="none",
    save_strategy="steps",
    no_cuda=not torch.cuda.is_available(),
    load_best_model_at_end=False
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# --- Step 10: Setup Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# --- Step 11: Resume from Checkpoint if Exists
last_checkpoint = None
if os.path.isdir(checkpoint_dir):
    checkpoints = [os.path.join(checkpoint_dir, ckpt) for ckpt in os.listdir(checkpoint_dir) if ckpt.startswith('checkpoint-')]
    if checkpoints:
        last_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"✅ Resuming training from checkpoint: {last_checkpoint}")
    else:
        print("ℹ️ No checkpoints found, starting fresh.")
else:
    print("ℹ️ No checkpoint folder found, starting fresh.")

# --- Step 12: Train Model
trainer.train(resume_from_checkpoint=last_checkpoint)

# --- Step 13: Save Final Model and Tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("🎯 Fine-tuning complete! Model saved to Google Drive.")
