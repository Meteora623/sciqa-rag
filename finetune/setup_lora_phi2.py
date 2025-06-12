# setup_lora_phi2.py

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch

# Step 1: model name
model_name = "microsoft/phi-2"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",         
    trust_remote_code=True
)
print("Model loaded successfully")

# Step 2: Loading Alpaca-style training data (approximately 50,000 instruction samples)
print("Loading Alpaca training data...")
dataset = load_dataset("tatsu-lab/alpaca")
train_data = dataset["train"]

# Print the first example
print("\n example:")
print(train_data[0])

# Step 3: LoRA configuration
print("\n Build LoRA configuration...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # standard transformer architecture
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Step 4: View trainable parameters
print("\n Trainable parameters statistics:")
model.print_trainable_parameters()

print("\n Environment and model architecture setup completed")
