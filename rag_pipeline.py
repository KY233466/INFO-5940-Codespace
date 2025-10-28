# rag_pipeline.py
from typing import List, Dict, Any
import os
from retrieval import format_docs, get_retriever
from chunking import split_named_texts
from vectorstore import build_ephemeral_vectorstore

def build_retriever_for_doc(doc_id: str, name: str, content: str, *, k: int = 20):
    """
    Build a retriever for a single document.
    Isolation is achieved by giving each doc its own persist_dir.
    """
    chunks = split_named_texts([(name, content)], chunk_size=200, chunk_overlap=0)

    per_doc_dir = os.path.join(".chroma", f"doc_{doc_id}")
    os.makedirs(per_doc_dir, exist_ok=True)

    vs = build_ephemeral_vectorstore(
        chunks,
        persist_dir=per_doc_dir,
    )
    return get_retriever(vs, k=k)

def retrieve_across(retrievers: List[Any], query: str, *, k_each: int = 10):
    hits = []
    for r in retrievers:
        try:
            hits.extend(r.invoke(query)[:k_each])
        except Exception:
            continue
    return hits

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