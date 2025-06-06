import gradio as gr
from rag_qa import answer_query_rag

gr.Interface(
    fn=answer_query_rag,
    inputs=gr.Textbox(lines=2, placeholder="Ask a scientific question..."),
    outputs="text",
    title="Scientific QA with RAG + Phi-2",
    description="Ask any science-related question. Answers are grounded in a custom corpus."
).launch()