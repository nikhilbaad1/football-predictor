"""Retrieval over unstructured text.

Structured facts are queried, not retrieved (ADR 0003). Only text with no schema
belongs here: what happened and why, not scorelines and dates.
"""

from fpp.rag.dense import dense_search, embed_corpus, embed_texts, load_vectors
from fpp.rag.hybrid import hybrid_search, reciprocal_rank_fusion
from fpp.rag.lexical import BM25Index, load_chunks, search, tokenize
from fpp.rag.wikipedia import chunk_text, fetch_page, ingest_pages, store_document

__all__ = [
    "BM25Index",
    "chunk_text",
    "dense_search",
    "embed_corpus",
    "embed_texts",
    "fetch_page",
    "hybrid_search",
    "ingest_pages",
    "load_chunks",
    "load_vectors",
    "reciprocal_rank_fusion",
    "search",
    "store_document",
    "tokenize",
]
