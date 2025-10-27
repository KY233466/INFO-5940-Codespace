# rag_pipeline.py
from typing import List, Dict, Any
from openai import OpenAI
from retrieval import format_docs
from chunking import split_named_texts
from vectorstore import build_ephemeral_vectorstore
from retrieval import get_retriever

def get_docs (named_texts, prompt):
    chunks = split_named_texts(named_texts, chunk_size=200, chunk_overlap=0)
    vs = build_ephemeral_vectorstore(chunks, persist_dir=None)  # set a path to persist between reruns
    retriever = get_retriever(vs, k=20)
    return retriever.invoke(prompt)

def build_messages(question: str, retrieved_docs) -> List[Dict[str, str]]:
    context_text = format_docs(retrieved_docs)
    system = {
        "role": "system",
        "content": (
            "You are a helpful assistant. "
            "Answer using ONLY the provided context. "
            "If the answer cannot be found, say you don't have enough information from the uploaded files."
        ),
    }
    user = {
        "role": "user",
        "content": f"Context:\n{context_text}\n\nQuestion: {question}",
    }
    return [system, user]