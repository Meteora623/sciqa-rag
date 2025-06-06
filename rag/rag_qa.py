import os
import torch
import faiss
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# 加载嵌入模型和 tokenizer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 加载 base 模型 + LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, "../finetune/lora-phi2-adapter-fast")
model.eval()

# 加载向量索引和文本块
index = faiss.read_index("faiss_index/docs.index")
with open("faiss_index/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# 问答函数
def answer_query_rag(query, top_k=3):
    # 召回相似段落
    query_embedding = embedding_model.encode([query])
    scores, indices = index.search(query_embedding, top_k)
    context = "\n\n".join([chunks[i] for i in indices[0]])[:1500]  # 控制 context 长度字符数

    # 改进后的 Prompt
    prompt = f"""You are a helpful scientific assistant. Use the context below to answer the question **in your own words**, clearly summarizing the key idea.

### Context:
{context}

### Question:
{query}

### Answer:"""

    # Token 限制（2048 总长 - 256 回答 = 最多输入 1792）
    max_input_tokens = 1792
    tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    input_ids = tokens["input_ids"].to(model.device)
    attention_mask = tokens["attention_mask"].to(model.device)

    # 推理
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id
        )

    # 只取生成的新 token
    generated = output_ids[0][input_ids.shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    answer = answer.split("\n### Question:")[0].strip()
    return answer

# 示例调用
# if __name__ == "__main__":
#     question = "What is the purpose of attention in transformers?"
#     print("Question:", question)
#     print("Answer:", answer_query_rag(question))

if __name__ == "__main__":
    while True:
        query = input("🔍 Enter your scientific question (or type 'exit'): ")
        if query.strip().lower() == "exit":
            break
        print("Answer:", answer_query_rag(query))
        print("=" * 80)