from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank-bm25 not installed. BM25 reranking disabled.")

def rerank(query: str, results: list[dict], top_k: int = 6) -> list[dict]:
    """
    Rerank search results using BM25.
    If rank-bm25 is not available, returns the original list.
    """
    if not BM25_AVAILABLE or not results:
        return results[:top_k]

    # Build corpus from title + snippet
    corpus = [
        f"{r.get('title', '')} {r.get('snippet', '')}".lower().split()
        for r in results
    ]
    
    try:
        bm25 = BM25Okapi(corpus)
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)
        
        # Attach scores to results
        for r, score in zip(results, scores):
            r["bm25_score"] = score
            
        # Sort by BM25 score
        sorted_results = sorted(results, key=lambda x: x["bm25_score"], reverse=True)
        return sorted_results[:top_k]
    except Exception as e:
        logger.error(f"BM25 reranking error: {e}")
        return results[:top_k]
