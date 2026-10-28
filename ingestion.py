"""
ingestion.py
Multi-modal ingestion for RAG Architect.

Supported inputs:
    PDF   -> PyMuPDF (fitz)
    CSV   -> pandas (summarised as shape, columns and sample rows)
    TXT   -> plain read
    Image -> pytesseract OCR (a lighter choice than CLIP for the free Spaces tier;
             swap CLIP in inside _read_image if needed)

Every extractor returns a normalized dict:
    {"filename": str, "modality": "pdf" | "csv" | "txt" | "image" | "error", "text": str}
"""

from __future__ import annotations

import os
from typing import Dict, Any, List

import pandas as pd


def _read_txt(path: str) -> Dict[str, Any]:
    with open(path, "r", errors="ignore") as f:
        text = f.read()
    return {
        "filename": os.path.basename(path),
        "modality": "txt",
        "text": text,
    }


def _read_pdf(path: str) -> Dict[str, Any]:
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    pages_text: List[str] = []
    for page in doc:
        pages_text.append(page.get_text())
    doc.close()
    text = "\n".join(pages_text)
    return {
        "filename": os.path.basename(path),
        "modality": "pdf",
        "text": text,
    }


def _read_csv(path: str) -> Dict[str, Any]:
    df = pd.read_csv(path, on_bad_lines="skip")
    # Build a short textual summary an LLM can reason over instead of
    # dumping the entire (potentially huge) table.
    sample = df.head(10).to_string(index=False)
    text = (
        f"CSV with {df.shape[0]} rows and {df.shape[1]} columns.\n"
        f"Columns: {', '.join(map(str, df.columns))}\n"
        f"Sample rows:\n{sample}"
    )
    return {
        "filename": os.path.basename(path),
        "modality": "csv",
        "text": text,
    }


def _read_image(path: str) -> Dict[str, Any]:
    from PIL import Image
    import pytesseract

    img = Image.open(path)
    try:
        text = pytesseract.image_to_string(img)
    except Exception as e:  # tesseract binary missing, etc.
        text = f"[OCR unavailable: {e}]"
    return {
        "filename": os.path.basename(path),
        "modality": "image",
        "text": text,
    }


EXTENSION_MAP = {
    ".pdf": _read_pdf,
    ".csv": _read_csv,
    ".txt": _read_txt,
    ".png": _read_image,
    ".jpg": _read_image,
    ".jpeg": _read_image,
    ".webp": _read_image,
}


def ingest_file(path: str) -> Dict[str, Any]:
    """Route a single uploaded file to the correct extractor."""
    ext = os.path.splitext(path)[1].lower()
    handler = EXTENSION_MAP.get(ext)
    if handler is None:
        raise ValueError(f"Unsupported file type: {ext}")
    return handler(path)


def ingest_files(paths: List[str]) -> List[Dict[str, Any]]:
    """Ingest multiple files, skipping ones that fail, collecting errors."""
    results = []
    for p in paths:
        if not p:
            continue
        try:
            results.append(ingest_file(p))
        except Exception:
            results.append({"filename": os.path.basename(p), "modality": "error", "text": ""})
    return results


def build_corpus_summary(ingested: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    """
    Merge multiple ingested documents into a single bounded text blob
    used as context for domain detection and the four agents.
    """
    parts = []
    for doc in ingested:
        if doc["modality"] == "error":
            continue
        snippet = doc["text"][:1500]
        parts.append(f"--- {doc['filename']} ({doc['modality']}) ---\n{snippet}")
    combined = "\n\n".join(parts)
    return combined[:max_chars]
