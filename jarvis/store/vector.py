"""
ChromaDB vector store for RAG.
Handles document chunking, indexing, and semantic retrieval.
"""

from __future__ import annotations

import os
import uuid
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")

# Use a default embedding function (sentence-transformers)
# Note: In a production environment, you might use an Ollama-based embedding function
default_ef = embedding_functions.DefaultEmbeddingFunction()

_client = None

def get_chroma_client():
    global _client
    if _client is None:
        if not os.path.exists(CHROMA_DIR):
            os.makedirs(CHROMA_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=True))
    return _client

async def add_to_vector_store(session_id: str, url: str, text: str, metadata: dict = None):
    """
    Chunks text and adds it to the session-specific collection in ChromaDB.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=f"session_{session_id.replace('-', '_')}",
        embedding_function=default_ef
    )
    
    # Simple semantic chunking (by paragraph or fixed size)
    chunks = _chunk_text(text, chunk_size=500, overlap=50)
    
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {**(metadata or {}), "url": url, "session_id": session_id} 
        for _ in chunks
    ]
    
    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )

async def query_vector_store(session_id: str, query: str, n_results: int = 5) -> list[dict]:
    """
    Queries the session-specific collection for semantically relevant chunks.
    """
    client = get_chroma_client()
    try:
        collection = client.get_collection(
            name=f"session_{session_id.replace('-', '_')}",
            embedding_function=default_ef
        )
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format results for the agent
        formatted = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                formatted.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else 0
                })
        return formatted
    except Exception:
        return []

def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks of approx. chunk_size characters."""
    if not text:
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
