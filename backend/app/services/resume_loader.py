"""Resume parsing and text extraction utilities.

Supports multiple file formats:
- PDF: Extracted using PyPDFLoader from LangChain
- Plain text: Direct parsing (txt, docx, etc.)
- Raw text: Direct input
"""
from __future__ import annotations

from pathlib import Path
import tempfile

from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from app.utils.text import clean_text


async def load_resume_text_from_upload(upload: UploadFile) -> tuple[str, str | None]:
    """
    Parse resume from file upload.
    
    Process:
    1. Detect file type from extension
    2. For PDF: Write to temp file, use PyPDFLoader
    3. For text: Decode bytes directly
    4. Clean and return extracted text with original filename
    
    Args:
        upload (UploadFile): FastAPI file upload object
    
    Returns:
        tuple: (cleaned_text, original_filename)
    """
    suffix = Path(upload.filename or "resume.txt").suffix.lower()
    data = await upload.read()

    if suffix == ".pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        text = "\n\n".join(doc.page_content for doc in docs)
        return clean_text(text), upload.filename

    raw_text = data.decode("utf-8", errors="ignore")
    return clean_text(raw_text), upload.filename


def load_resume_text_from_plain_text(text: str) -> tuple[str, str | None]:
    """
    Parse resume from plain text input.
    
    Args:
        text (str): Raw resume text from form input
    
    Returns:
        tuple: (cleaned_text, None)
    """
    return clean_text(text), None


def load_text_from_file_path(file_path: str) -> str:
    """
    Parse resume from file system path.
    
    Process:
    1. Detect file type from extension
    2. Use appropriate LangChain loader
    3. Combine and clean all pages/content
    
    Args:
        file_path (str): Full path to file
    
    Returns:
        str: Cleaned text content
    """
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        return clean_text("\n\n".join(doc.page_content for doc in docs))
    loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()
    return clean_text("\n\n".join(doc.page_content for doc in docs))
