# setup_lora_phi2.py

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch

# Step 1: 模型名称（轻量模型，适合本地显卡）
model_name = "microsoft/phi-2"

print("正在加载 tokenizer 和模型...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",         # 自动将模型加载到GPU
    trust_remote_code=True
)
print("模型加载成功")

# Step 2: 加载 Alpaca-style 训练数据（约5万条instruction样本）
print("加载 Alpaca 数据集...")
dataset = load_dataset("tatsu-lab/alpaca")
train_data = dataset["train"]

# 打印前1条示例
print("\n 示例数据:")
print(train_data[0])

# Step 3: LoRA 配置
print("\n 构建 LoRA 配置...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Phi-2中是标准transformer结构
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 应用 LoRA 到模型
model = get_peft_model(model, lora_config)

# Step 4: 查看可训练参数
print("\n 可训练参数统计:")
model.print_trainable_parameters()

print("\n 环境与模型结构准备完成")