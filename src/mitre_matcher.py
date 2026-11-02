from typing import Any, Dict, List

from mitre_vector_store import MitreVectorStore


DEFAULT_TOP_K = 10
DEFAULT_MIN_SIMILARITY = 0.20


class MitreCandidateRetriever:
    def __init__(self, vector_store: MitreVectorStore = None):
        self.store = vector_store or MitreVectorStore()

    def retrieve_candidates(
        self,
        alert: Dict[str, Any],
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
    ) -> List[Dict[str, Any]]:
        """
        Given a standardized alert dict, return a ranked list of candidate techniques.

        Expected alert keys:
          - embedding_text (required)
          - alert_id (for logging)
        """
        query_text = alert.get("embedding_text")
        if not query_text:
            raise ValueError(
                f"Alert {alert.get('alert_id')} has no 'embedding_text' field."
            )

        results = self.store.collection.query(
            query_texts=[query_text],
            n_results=top_k,
        )

        candidates = []
        if not results["ids"] or not results["ids"][0]:
            return candidates

        for attack_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = max(0.0, min(1.0, 1.0 - dist))
            if similarity < min_similarity:
                continue
            candidates.append(
                {
                    "attack_id": attack_id,
                    "name": meta["name"],
                    "is_subtechnique": bool(meta["is_subtechnique"]),
                    "parent_id": meta.get("parent_id") or None,
                    "tactics": meta.get("tactics", "").split(",")
                    if meta.get("tactics")
                    else [],
                    "description_snippet": doc[:300],
                    "similarity": round(similarity, 4),
                }
            )

        candidates.sort(key=lambda c: c["similarity"], reverse=True)
        for i, c in enumerate(candidates, start=1):
            c["rank"] = i

        return candidates


if __name__ == "__main__":
    sample_alert = {
        "alert_id": "WEB-1006",
        "embedding_text": (
            "Alert: Web Login Brute Force | Severity: HIGH | "
            "Description: Trigger when one source IP produces at least 20 failed login "
            "attempts within 5 minutes. | "
            "Source: 192.168.1.55 | Target: /login | "
            "Evidence: Failed authentication threshold exceeded."
        ),
    }

    retriever = MitreCandidateRetriever()
    cands = retriever.retrieve_candidates(sample_alert, top_k=5)

    print(f"\n[CANDIDATES] Alert {sample_alert['alert_id']} -> {len(cands)} candidates:")
    for c in cands:
        print(
            f"  #{c['rank']} {c['attack_id']:12s} "
            f"sim={c['similarity']:.3f} "
            f"sub={c['is_subtechnique']} "
            f"name={c['name']}"
        )