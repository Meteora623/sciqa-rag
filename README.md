# SciQA-RAG

A scientific domain question-answering system powered by RAG (Retrieval-Augmented Generation) and LoRA fine-tuned Phi-2.

## Project Structure

sciqa-rag/
├── data/ # Input data: instruct, papers, and texts
├── finetune/ # LoRA fine-tuning for Phi-2
│ ├── setup_lora_phi2.py
│ ├── train_lora_phi2.py
│ └── lora-phi2-checkpoint*/ # Model checkpoints and adapters
├── rag/ # RAG pipeline for document indexing and QA
│ ├── extract_text.py
│ ├── build_index.py
│ ├── baseline_qa.py
│ ├── rag_qa.py
│ ├── web_demo.py
│ └── faiss_index/ # Stored FAISS vector index
├── inference_lora_phi2.py # Final integrated QA pipeline


## Features

- Finetune [Phi-2](https://huggingface.co/microsoft/phi-2) using [LoRA](https://github.com/microsoft/LoRA)
- Chunk scientific documents and build a local FAISS index
- Use Retrieval-Augmented Generation (RAG) for question answering
- Web-based demo interface via `Gradio`

## Usage
1. Prepare and Index the Data
python rag/extract_text.py         # Convert raw documents to chunks
python rag/build_index.py          # Build FAISS index

3. Fine-tune Phi-2 Model
python finetune/train_lora_phi2.py

4. Run the Web Demo
python rag/web_demo.py

You can test with scientific questions like:

"What is the main idea behind retrieval-augmented generation?"
"How does LoRA improve fine-tuning efficiency?"

