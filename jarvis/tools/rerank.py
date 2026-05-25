from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

_model = None

def get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading CrossEncoder model 'cross-encoder/ms-marco-MiniLM-L-6-v2'...")
            _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except ImportError:
            logger.warning("sentence-transformers not installed. Cross-encoder reranking disabled.")
            return None
    return _model

def cross_encode_rerank(query: str, results: list[dict], top_k: int = 4) -> list[dict]:
    """
    Rerank search results using a Cross-Encoder for high precision.
    """
    if not results:
        return []
        
    model = get_model()
    if not model:
        return results[:top_k]
        
    try:
        texts = [f"{r.get('title', '')}. {r.get('snippet', '')}" for r in results]
        scores = model.predict([(query, text) for text in texts])
        
        for r, score in zip(results, scores):
            r["cross_score"] = float(score)
            
        sorted_results = sorted(results, key=lambda x: x["cross_score"], reverse=True)
        return sorted_results[:top_k]
    except Exception as e:
        logger.error(f"Cross-encoder reranking error: {e}")
        return results[:top_k]
