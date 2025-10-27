# vectorstore.py
import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def _embeddings() -> OpenAIEmbeddings:
    # Works with your Cornell proxy if you export:
    #   export OPENAI_API_KEY=<API_KEY>
    #   export OPENAI_BASE_URL=https://api.ai.it.cornell.edu
    # Or pass via env vars in Codespaces settings.
    return OpenAIEmbeddings(
        model="openai.text-embedding-3-large",
        api_key=os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.ai.it.cornell.edu"),
    )

def build_ephemeral_vectorstore(
    docs: List[Document],
    persist_dir: Optional[str] = None,
) -> Chroma:
    """
    If persist_dir is provided, Chroma will persist between reruns.
    Otherwise, it's in-memory for the current run.
    """
    vs = Chroma.from_documents(
        documents=docs,
        embedding=_embeddings(),
        persist_directory=persist_dir,
    )
    return vs