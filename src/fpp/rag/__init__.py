"""Retrieval over unstructured text.

Structured facts are queried, not retrieved (ADR 0003). Only text with no schema
belongs here: what happened and why, not scorelines and dates.
"""

from fpp.rag.lexical import BM25Index, load_chunks, search, tokenize
from fpp.rag.wikipedia import chunk_text, fetch_page, ingest_pages, store_document

__all__ = [
    "BM25Index",
    "chunk_text",
    "fetch_page",
    "ingest_pages",
    "load_chunks",
    "search",
    "store_document",
    "tokenize",
]
