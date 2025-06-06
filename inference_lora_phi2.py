import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# 设定路径
base_model = "microsoft/phi-2"
adapter_path = "./finetune/lora-phi2-adapter-fast"  # 你的 LoRA adapter 目录

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 加载 base 模型 + LoRA adapter
base = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base, adapter_path)
model.eval()

# 推理函数
def generate_response(instruction: str, input_text: str = ""):
    if input_text:
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()


# 示例测试
if __name__ == "__main__":
    question = "What are three benefits of exercise?"
    print("Question:", question)
    print("Answer:", generate_response(question))