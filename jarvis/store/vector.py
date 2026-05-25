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
    await add_documents_to_collection(
        collection_name=f"session_{session_id.replace('-', '_')}",
        documents=_chunk_text(text),
        metadatas=[{**(metadata or {}), "url": url, "session_id": session_id}]
    )

async def add_documents_to_collection(collection_name: str, documents: list[str], metadatas: list[dict] = None):
    """
    Add a list of document chunks to a specified collection.
    If metadatas is a single dict, it's applied to all chunks.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=default_ef
    )
    
    ids = [str(uuid.uuid4()) for _ in documents]
    
    if metadatas and len(metadatas) == 1 and len(documents) > 1:
        final_metadatas = [metadatas[0] for _ in documents]
    else:
        final_metadatas = metadatas
        
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=final_metadatas
    )

async def delete_collection_items(collection_name: str, filter: dict):
    """Delete items from a collection based on metadata filter."""
    client = get_chroma_client()
    try:
        collection = client.get_collection(name=collection_name, embedding_function=default_ef)
        collection.delete(where=filter)
    except Exception:
        pass

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

class EpisodicMemory:
    async def store_session(self, session_id: str, goal: str, answer: str) -> None:
        """Index the final answer for future cross-session retrieval."""
        client = get_chroma_client()
        collection = client.get_or_create_collection(
            name="episodic",
            embedding_function=default_ef
        )
        
        # We index the full answer as a single "memory" (or chunk it if very long)
        # For simplicity, just one doc per session here.
        collection.add(
            documents=[f"Goal: {goal}\n\nAnswer: {answer}"],
            ids=[session_id],
            metadatas=[{"session_id": session_id, "goal": goal, "type": "research_result"}]
        )

    async def search(self, goal: str, n: int = 3) -> list[dict]:
        client = get_chroma_client()
        try:
            # We'll use a single 'episodic' collection for cross-session memory
            collection = client.get_or_create_collection(
                name="episodic",
                embedding_function=default_ef
            )
            if collection.count() == 0:
                return []
            
            results = collection.query(
                query_texts=[goal],
                n_results=n
            )
            
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
        
episodic_memory = EpisodicMemory()

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
