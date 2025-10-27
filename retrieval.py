# retrieval.py
from typing import Dict, Any
from langchain_chroma import Chroma

def get_retriever(vs: Chroma, k: int = 20):
    # Same default `k` as shown in lecture notes
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})

def format_docs(docs) -> str:
    # Join doc contents; include sources to help debug / cite
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        parts.append(f"[{src}] {d.page_content}")
    return "\n\n".join(parts)