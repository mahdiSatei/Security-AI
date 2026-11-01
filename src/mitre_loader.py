import json
import os
import urllib.request
from typing import Any, Dict, List, Optional

STIX_URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
LOCAL_FILE = os.path.join("data", "enterprise-attack.json")


class MitreDataManager:
    def __init__(self, file_path: str = LOCAL_FILE, auto_download: bool = True):
        self.file_path = file_path
        if not os.path.exists(self.file_path) and auto_download:
            self._download_stix_data()

        self.techniques: Dict[str, Dict[str, Any]] = {}
        self._stix_id_to_mitre_id: Dict[str, str] = {}
        self._load_and_parse()

    def _download_stix_data(self):
        """دانلود داده‌های رسمی STIX مایتر با نمایش درصد پیشرفت"""
        print("[INFO] STIX file not found locally. Downloading from GitHub...")
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100.0, (downloaded / total_size) * 100)
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r[DOWNLOADING] {mb_down:.1f}MB / {mb_total:.1f}MB ({percent:.1f}%)", end="")
            else:
                print(f"\r[DOWNLOADING] {downloaded / (1024 * 1024):.1f}MB", end="")

        try:
            urllib.request.urlretrieve(STIX_URL, self.file_path, reporthook=progress_hook)
            print(f"\n[OK] Successfully downloaded STIX data to: {self.file_path}")
        except Exception as e:
            if os.path.exists(self.file_path):
                os.remove(self.file_path)
            raise RuntimeError(f"Failed to download MITRE STIX data: {e}")

    def _load_and_parse(self):
        """پارس کردن تکنیک‌ها، زیرتکنیک‌ها و استخراج والدها"""
        print("[INFO] Loading and parsing STIX data (this may take a few seconds)...")
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        subtechnique_relations = []

        # Pass 1: استخراج تکنیک‌های فعال
        for obj in data.get("objects", []):
            obj_type = obj.get("type")

            if obj_type == "attack-pattern":
                if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
                    continue

                mitre_id = None
                for ref in obj.get("external_references", []):
                    if ref.get("source_name") == "mitre-attack":
                        mitre_id = ref.get("external_id")
                        break

                if mitre_id:
                    stix_id = obj["id"]
                    is_sub = obj.get("x_mitre_is_subtechnique", False)

                    tech_info = {
                        "id": mitre_id,
                        "name": obj.get("name"),
                        "description": obj.get("description", ""),
                        "is_subtechnique": is_sub,
                        "parent_id": None,
                        "platforms": obj.get("x_mitre_platforms", []),
                        "kill_chain_phases": [
                            p.get("phase_name") for p in obj.get("kill_chain_phases", [])
                        ],
                    }

                    self.techniques[mitre_id] = tech_info
                    self._stix_id_to_mitre_id[stix_id] = mitre_id

            # استخراج روابط زیرتکنیک به تکنیک والد
            elif obj_type == "relationship" and obj.get("relationship_type") == "subtechnique-of":
                source_stix = obj.get("source_ref")
                target_stix = obj.get("target_ref")
                subtechnique_relations.append((source_stix, target_stix))

        # Pass 2: متصل کردن شناسه والد به هر زیرتکنیک
        for sub_stix, parent_stix in subtechnique_relations:
            sub_mitre_id = self._stix_id_to_mitre_id.get(sub_stix)
            parent_mitre_id = self._stix_id_to_mitre_id.get(parent_stix)

            if sub_mitre_id and parent_mitre_id and sub_mitre_id in self.techniques:
                self.techniques[sub_mitre_id]["parent_id"] = parent_mitre_id

        print(f"[OK] Parsing complete! Loaded {len(self.techniques)} active techniques/sub-techniques.")

    def get_technique(self, mitre_id: str) -> Optional[Dict[str, Any]]:
        return self.techniques.get(mitre_id.strip().upper())

    def is_valid_id(self, mitre_id: str) -> bool:
        return mitre_id.strip().upper() in self.techniques

    def get_all_techniques(self) -> List[Dict[str, Any]]:
        return list(self.techniques.values())


if __name__ == "__main__":
    manager = MitreDataManager()
    print("Sample technique T1059.001:", manager.get_technique("T1059.001")["name"])