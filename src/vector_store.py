import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.utils import embedding_functions


class AlertVectorStore:
    def __init__(
        self,
        persist_dir: str = "data/chroma_db",
        collection_name: str = "security_alerts",
        model_name: str = "all-MiniLM-L6-v2",
    ):
       
        self.persist_dir = persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_dir)

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def index_alerts(self, alerts: List[Dict[str, Any]]) -> None:
      
        ids = []
        documents = []
        metadatas = []

        for alert in alerts:
            ids.append(alert["alert_id"])
            documents.append(alert["embedding_text"])

            clean_meta = {
                "rule_name": alert.get("rule_name", ""),
                "severity": alert.get("severity", "unknown"),
                "risk_score": alert.get("risk_score") or 0,
                "source_ip": alert.get("source_ip") or "",
                "destination_ip": alert.get("destination_ip") or "",
                "target_resource": alert.get("target_resource") or "",
            }
            metadatas.append(clean_meta)

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        print(f"✅ تعداد {len(ids)} هشدار با موفقیت امبد و در ChromaDB ذخیره شدند.")

    def search(
        self,
        query: str,
        n_results: int = 5,
        severity_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
       
        where_clause = None
        if severity_filter:
            where_clause = {"severity": severity_filter.lower().strip()}

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause,
        )

        formatted_results = []
        if not results["ids"] or not results["ids"][0]:
            return formatted_results

        for rank, (alert_id, doc, meta, distance) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ),
            start=1,
        ):
  
            similarity_score = max(0.0, min(1.0, 1.0 - distance))

            formatted_results.append(
                {
                    "rank": rank,
                    "alert_id": alert_id,
                    "similarity_score": round(similarity_score, 4),
                    "rule_name": meta.get("rule_name"),
                    "severity": meta.get("severity"),
                    "source_ip": meta.get("source_ip"),
                    "target_resource": meta.get("target_resource"),
                    "text": doc,
                }
            )

        return formatted_results