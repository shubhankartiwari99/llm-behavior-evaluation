from app.intelligence.embedding_engine import EmbeddingEngine

embedding_engine = EmbeddingEngine()

def ambiguity_score(text: str) -> float:
    hedges = [
        "maybe", "might", "could", "perhaps",
        "i think", "possibly", "it seems"
    ]
    count = 0
    lower = text.lower()
    for h in hedges:
        count += lower.count(h)
    return min(count / 5.0, 1.0)  # normalized cap

def evaluate_dual_plane(det_output: str, ent_output: str, det_tokens: int, ent_tokens: int):

    length_delta = abs(len(det_output) - len(ent_output))
    token_delta = abs(det_tokens - ent_tokens)

    normalized_length = min(length_delta / 500.0, 1.0)
    normalized_token = min(token_delta / 150.0, 1.0)

    embedding_similarity = embedding_engine.similarity(det_output, ent_output)
    semantic_instability = 1 - embedding_similarity

    ambiguity = ambiguity_score(ent_output)

    instability = (
        0.30 * normalized_length
        + 0.20 * normalized_token
        + 0.35 * semantic_instability
        + 0.15 * ambiguity
    )

    instability = min(instability, 1.0)
    confidence = 1 - instability
    escalate = instability > 0.38

    return {
        "instability": round(instability, 3),
        "confidence": round(confidence, 3),
        "escalate": escalate,
        "embedding_similarity": round(embedding_similarity, 3),
        "ambiguity": round(ambiguity, 3)
    }
