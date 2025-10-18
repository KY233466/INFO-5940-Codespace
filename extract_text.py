# text_extractor.py
from __future__ import annotations
import io
from pathlib import Path
from typing import Optional

def _extract_text_from_txt(file_bytes: bytes, encoding: Optional[str] = None) -> str:
    """
    Try UTF-8 first; if it fails, fall back to latin-1 (broad but safe).
    """
    if encoding:
        try:
            return file_bytes.decode(encoding, errors="strict")
        except Exception:
            pass

    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # very permissive fallback to avoid crashes on odd encodings
        return file_bytes.decode("latin-1", errors="ignore")


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Prefer pdfminer.six for better extraction accuracy; fall back to PyPDF2 if needed.
    """
    # Try pdfminer.six
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text
        with io.BytesIO(file_bytes) as f:
            text = pdfminer_extract_text(f) or ""
        # Normalize weird hyphenation artifacts a bit
        return text.replace("\r", "")
    except Exception:
        pass

    # Fallback: PyPDF2
    try:
        import PyPDF2
        text_chunks = []
        with io.BytesIO(file_bytes) as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text_chunks.append(page_text)
        return "\n".join(text_chunks)
    except Exception as e:
        raise RuntimeError(
            "Unable to extract text from PDF (pdfminer and PyPDF2 both failed)."
        ) from e


def extract_text(uploaded_file) -> str:
    """
    Streamlit's `uploaded_file` can be treated like a file-like object.
    We detect by extension and/or MIME type, read bytes once, and route to the right handler.
    """
    # Read the bytes once (Streamlit's UploadedFile is a SpooledTemporaryFile-like object)
    file_bytes = uploaded_file.read()

    # Try to detect by extension first
    suffix = Path(uploaded_file.name).suffix.lower()
    mime = getattr(uploaded_file, "type", "")  # e.g., "application/pdf" or "text/plain"

    is_pdf = suffix == ".pdf" or mime == "application/pdf"
    is_txt = suffix in {".txt", ".text"} or mime.startswith("text/")

    if is_pdf:
        return _extract_text_from_pdf(file_bytes).strip()

    if is_txt:
        return _extract_text_from_txt(file_bytes).strip()

    # Last-resort guess: try text decode, then PDF
    try:
        return _extract_text_from_txt(file_bytes).strip()
    except Exception:
        return _extract_text_from_pdf(file_bytes).strip()
