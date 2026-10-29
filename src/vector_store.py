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

            # اضافه کردن فیلدهای متادیتا شامل Hostname و Severity برای فیلترینگ
            clean_meta = {
                "rule_name": alert.get("rule_name", ""),
                "severity": str(alert.get("severity", "unknown")).lower(),
                "hostname": str(alert.get("hostname") or "unknown_host"),
                "source_ip": str(alert.get("source_ip") or ""),
                "destination_ip": str(alert.get("destination_ip") or ""),
                "target_resource": str(alert.get("target_resource") or ""),
            }
            metadatas.append(clean_meta)

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        print(f"✅ تعداد {len(ids)} هشدار با موفقیت ایندکس شدند.")

    def search(
        self,
        query: str,
        n_results: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        جستجوی معنایی با پشتیبانی از فیلترهای دلخواه (تکی یا ترکیبی):
        مثال:
            filters={"severity": "high"}
            filters={"severity": "high", "source_ip": "192.168.1.50"}
        """
        where_clause = None

        if filters:
            # پاک‌سازی مقادیر خالی
            active_filters = [{k: str(v).lower()} for k, v in filters.items() if v]

            if len(active_filters) == 1:
                where_clause = active_filters[0]
            elif len(active_filters) > 1:
                # شرط AND در Chroma برای چند فیلتر همزمان
                where_clause = {"$and": active_filters}

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
                    "hostname": meta.get("hostname"),
                    "source_ip": meta.get("source_ip"),
                    "target_resource": meta.get("target_resource"),
                    "text": doc,
                }
            )

        return formatted_results