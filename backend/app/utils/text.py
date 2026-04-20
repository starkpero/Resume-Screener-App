"""Text processing utilities for resume and job description handling.

This module provides:
- Text cleaning and normalization
- Email and name extraction via regex
- Document chunking for vector embeddings
- Deduplication of extracted fields
"""
from __future__ import annotations

import re
from typing import Iterable

from langchain_core.documents import Document

# Regex patterns for extraction
EMAIL_RE = re.compile(r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})')
NAME_HINT_RE = re.compile(r'(?i)\b(name|candidate)\s*[:\-]\s*(.+)')


def clean_text(text: str) -> str:
    """
    Normalize and clean raw resume/JD text.
    
    Process:
    1. Remove null bytes (can appear in corrupted PDFs)
    2. Normalize line endings (Windows/Unix)
    3. Collapse multiple spaces/tabs to single space
    4. Consolidate multiple blank lines to max 2
    5. Strip leading/trailing whitespace
    
    Args:
        text (str): Raw unformatted text
    
    Returns:
        str: Cleaned and normalized text
    """
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_email(text: str) -> str | None:
    """
    Extract email address from text using regex.
    
    Pattern: [local]@[domain].[tld]
    Returns first match found
    
    Args:
        text (str): Text to search
    
    Returns:
        str | None: First email address found, or None
    """
    match = EMAIL_RE.search(text or "")
    return match.group(1) if match else None


def split_text_for_docs(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Document]:
    """
    Split text into chunks for vector embedding and semantic search.
    
    Uses RecursiveCharacterTextSplitter from LangChain:
    - Chunk size: 1000 characters (configurable)
    - Overlap: 150 characters (preserves context across chunks)
    - Strategy: Splits on sentence/paragraph boundaries first
    
    Overlapping chunks ensure semantic search finds relevant sections
    even if the query spans chunk boundaries.
    
    Args:
        text (str): Full document text
        chunk_size (int): Target chunk size in characters
        chunk_overlap (int): Overlap between consecutive chunks
    
    Returns:
        list[Document]: LangChain Document objects with metadata
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents([text])


def normalize_list(items: Iterable[str]) -> list[str]:
    """
    Deduplicate and normalize string list.
    
    Process:
    1. Normalize whitespace within each item
    2. Remove duplicates (case-insensitive comparison)
    3. Remove empty strings
    4. Preserve original casing of first occurrence
    
    This handles LLM output deduplication where similar items might be
    represented differently (e.g., "Python" vs "python", "  Python  ")
    
    Args:
        items (Iterable[str]): List of strings from LLM output
    
    Returns:
        list[str]: Deduplicated, normalized list
    """
    unique: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = re.sub(r"\s+", " ", item).strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique


def maybe_extract_name_from_header(text: str) -> str | None:
    """
    Extract candidate name from resume header section.
    
    Heuristics (in order):
    1. Look for "Name:" or "Name -" pattern in first 10 lines
    2. If not found, try first non-empty line with:
       - 4 words or fewer (typical name format)
       - No digits (filters out "123 Main St" addresses)
       - No email symbols
    3. Fallback: None
    
    Args:
        text (str): Resume text
    
    Returns:
        str | None: Extracted name (max 80 chars), or None
    """
    first_lines = "\n".join((text or "").splitlines()[:10])
    match = NAME_HINT_RE.search(first_lines)
    if match:
        candidate = match.group(2).strip()
        return candidate[:80] if candidate else None
    for line in (line.strip() for line in first_lines.splitlines()):
        if not line:
            continue
        if len(line.split()) <= 4 and not any(ch.isdigit() for ch in line) and "@" not in line:
            return line[:80]
    return None
