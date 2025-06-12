import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Set path
text_folder = "../data/texts"
index_path = "faiss_index"
embedding_model_name = "all-MiniLM-L6-v2"

# Load embedding model (Sentence-Transformers)
model = SentenceTransformer(embedding_model_name)

# Collect all text blocks
documents = []
file_names = os.listdir(text_folder)
for file in file_names:
    with open(os.path.join(text_folder, file), "r", encoding="utf-8") as f:
        text = f.read()
        chunks = [chunk.strip() for chunk in text.split("\n\n") if len(chunk.strip()) > 50]
        documents.extend(chunks)

# Generate embeddings
print(f"Generating embedding vectors for {len(documents)} text chunks...")
embeddings = model.encode(documents, show_progress_bar=True)

# Build and save FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

os.makedirs(index_path, exist_ok=True)
faiss.write_index(index, os.path.join(index_path, "docs.index"))

with open(os.path.join(index_path, "chunks.pkl"), "wb") as f:
    pickle.dump(documents, f)

print("Vector index construction completed and saved to ./rag/faiss_index/")
