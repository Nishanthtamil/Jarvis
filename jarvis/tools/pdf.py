"""
PDF extraction using PyMuPDF (fitz).
Handles local files and remote PDF URLs.
"""

from __future__ import annotations

import io
import os

import fitz  # PyMuPDF
import httpx

from llm.client import truncate_to_tokens, count_tokens

MAX_TOKENS_PER_SOURCE = int(os.getenv("MAX_TOKENS_PER_SOURCE", "1500"))


async def extract_pdf_url(url: str) -> dict:
    """Download and extract text from a remote PDF."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            pdf_bytes = resp.content

        return _extract_from_bytes(url, pdf_bytes)
    except Exception as e:
        return {"url": url, "markdown": "", "token_count": 0, "ok": False, "error": str(e)}


def extract_pdf_file(path: str) -> dict:
    """Extract text from a local PDF file."""
    try:
        with open(path, "rb") as f:
            pdf_bytes = f.read()
        return _extract_from_bytes(path, pdf_bytes)
    except Exception as e:
        return {"url": path, "markdown": "", "token_count": 0, "ok": False, "error": str(e)}


def _extract_from_bytes(source: str, pdf_bytes: bytes) -> dict:
    """Core extraction logic — iterates pages and collects text."""
    doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    pages = []
    for page_num in range(min(doc.page_count, 50)):  # cap at 50 pages
        page = doc[page_num]
        text = page.get_text("text")  # plain text extraction
        if text.strip():
            pages.append(f"[Page {page_num + 1}]\n{text.strip()}")
    doc.close()

    full_text = "\n\n".join(pages)
    truncated = truncate_to_tokens(full_text, MAX_TOKENS_PER_SOURCE)

    return {
        "url": source,
        "markdown": truncated,
        "token_count": count_tokens(truncated),
        "ok": True,
        "error": None,
    }


def is_pdf_url(url: str) -> bool:
    return url.lower().endswith(".pdf") or "pdf" in url.lower()
