import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 设定路径
text_folder = "../data/texts"
index_path = "faiss_index"
embedding_model_name = "all-MiniLM-L6-v2"

# 加载嵌入模型（Sentence-Transformers）
model = SentenceTransformer(embedding_model_name)

# 收集所有文本块
documents = []
file_names = os.listdir(text_folder)
for file in file_names:
    with open(os.path.join(text_folder, file), "r", encoding="utf-8") as f:
        text = f.read()
        chunks = [chunk.strip() for chunk in text.split("\n\n") if len(chunk.strip()) > 50]
        documents.extend(chunks)

# 生成嵌入
print(f"📄 正在为 {len(documents)} 段文本生成嵌入向量...")
embeddings = model.encode(documents, show_progress_bar=True)

# 构建并保存 FAISS 索引
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

os.makedirs(index_path, exist_ok=True)
faiss.write_index(index, os.path.join(index_path, "docs.index"))

with open(os.path.join(index_path, "chunks.pkl"), "wb") as f:
    pickle.dump(documents, f)

print("向量索引构建完成，保存于 ./rag/faiss_index/")