# chunking.py
from typing import Iterable, List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_named_texts(
    items: Iterable[Tuple[str, str]],
    chunk_size: int = 200,
    chunk_overlap: int = 0,
) -> List[Document]:
    """
    items: iterable of (name, text) pairs
    returns: list of LangChain Document chunks with metadata {"source": name}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs: List[Document] = []
    for name, text in items:
        if not text or not text.strip():
            continue
        # Make one big Document first (per file), then split
        base = Document(page_content=text, metadata={"source": name})
        docs.extend(splitter.split_documents([base]))
    return docs