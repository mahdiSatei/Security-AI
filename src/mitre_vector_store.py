import json
import os
from pathlib import Path
from typing import Any, Dict

import chromadb
from chromadb.utils import embedding_functions

DATA_DIR = "data"
VALID_TECHNIQUES_PATH = os.path.join(DATA_DIR, "valid_techniques.json")


class MitreVectorStore:
    def __init__(
        self,
        persist_dir: str = "data/chroma_db",
        collection_name: str = "mitre_techniques",
        model_name: str = "all-MiniLM-L6-v2",
    ):
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

    def count(self) -> int:
        return self.collection.count()

    @staticmethod
    def _build_technique_text(info: Dict[str, Any]) -> str:
        parts = [
            f"Technique: {info['name']}",
            f"ID: {info['attack_id']}",
        ]
        if info.get("is_subtechnique"):
            parts.append(f"Sub-technique of: {info.get('parent_id', 'unknown')}")
        if info.get("description"):
            parts.append(f"Description: {info['description']}")
        if info.get("platforms"):
            parts.append(f"Platforms: {', '.join(info['platforms'])}")
        if info.get("kill_chain_phases"):
            parts.append(f"Tactics: {', '.join(info['kill_chain_phases'])}")
        return " | ".join(parts)

    def index_techniques(self, force_reindex: bool = False) -> None:
        if not os.path.exists(VALID_TECHNIQUES_PATH):
            raise FileNotFoundError(
                f"{VALID_TECHNIQUES_PATH} not found. Run build_valid_techniques.py first."
            )

        with open(VALID_TECHNIQUES_PATH, "r", encoding="utf-8") as f:
            techniques = json.load(f)

        if force_reindex:
            existing = self.collection.get()["ids"]
            if existing:
                self.collection.delete(ids=existing)
                print("[!] MITRE collection cleared. Re-indexing...")

        existing_ids = set(self.collection.get()["ids"])
        new_items = {k: v for k, v in techniques.items() if k not in existing_ids}

        if not new_items:
            print(f"[*] MITRE collection is already up-to-date ({self.count()} indexed).")
            return

        ids, documents, metadatas = [], [], []
        for attack_id, info in new_items.items():
            ids.append(attack_id)
            documents.append(self._build_technique_text(info))
            metadatas.append(
                {
                    "attack_id": attack_id,
                    "name": info["name"],
                    "is_subtechnique": info["is_subtechnique"],
                    "parent_id": info.get("parent_id") or "",
                    "tactics": ",".join(info.get("kill_chain_phases", [])),
                }
            )

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        print(f"[+] Successfully indexed {len(ids)} techniques. Total: {self.count()}")


if __name__ == "__main__":
    store = MitreVectorStore()
    store.index_techniques()