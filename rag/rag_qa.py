import os
import torch
import faiss
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# Load embedding model and tokenizer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load base model with LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, "../finetune/lora-phi2-adapter-fast")
model.eval()

# Load vector index and text blocks
index = faiss.read_index("faiss_index/docs.index")
with open("faiss_index/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Question-answering function
def answer_query_rag(query, top_k=3):
    # Retrieve similar passages
    query_embedding = embedding_model.encode([query])
    scores, indices = index.search(query_embedding, top_k)
    context = "\n\n".join([chunks[i] for i in indices[0]])[:1500]  # Control context length by character count

    # Prompt
    prompt = f"""You are a helpful scientific assistant. Use the context below to answer the question **in your own words**, clearly summarizing the key idea.

### Context:
{context}

### Question:
{query}

### Answer:"""

    # Token limit (2048 total - 256 for answer = maximum input 1792)
    max_input_tokens = 1792
    tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    input_ids = tokens["input_ids"].to(model.device)
    attention_mask = tokens["attention_mask"].to(model.device)

    # Inference
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

    # Only extract the newly generated tokens
    generated = output_ids[0][input_ids.shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    answer = answer.split("\n### Question:")[0].strip()
    return answer

# Example invocation
# if __name__ == "__main__":
#     question = "What is the purpose of attention in transformers?"
#     print("Question:", question)
#     print("Answer:", answer_query_rag(question))

if __name__ == "__main__":
    while True:
        query = input("Enter your scientific question (or type 'exit'): ")
        if query.strip().lower() == "exit":
            break
        print("Answer:", answer_query_rag(query))
        print("=" * 80)
