# rag_pipeline.py
from typing import List, Dict, Any
from retrieval import format_docs
from chunking import split_named_texts
from vectorstore import build_ephemeral_vectorstore
from retrieval import get_retriever

def build_retriever_for_doc(name: str, content: str, *, k: int = 20):
    """
    Build a retriever for a single document.
    Called once per upload; the caller should cache/reuse this object.
    """
    chunks = split_named_texts([(name, content)], chunk_size=200, chunk_overlap=0)
    vs = build_ephemeral_vectorstore(chunks, persist_dir=None)
    return get_retriever(vs, k=k)

def retrieve_across(retrievers: List[Any], query: str, *, k_each: int = 10):
    """
    Query multiple retrievers and merge results.
    Simple concat; you can add ranking/dup filtering later if desired.
    """
    hits = []
    for r in retrievers:
        try:
            hits.extend(r.invoke(query)[:k_each])
        except Exception:
            # If any single retriever fails, keep going
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