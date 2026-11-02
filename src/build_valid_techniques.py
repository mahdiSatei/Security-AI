import json
import os
from mitre_loader import MitreDataManager

DATA_DIR = "data"
OUTPUT_PATH = os.path.join(DATA_DIR, "valid_techniques.json")


def build_valid_techniques(output_path: str = OUTPUT_PATH) -> dict:
    print("[INFO] Initializing MitreDataManager...")
    manager = MitreDataManager()

    valid = {}
    for mitre_id, info in manager.techniques.items():
        valid[mitre_id] = {
            "attack_id": info["id"],
            "name": info["name"],
            "description": (info.get("description") or "")[:500],
            "is_subtechnique": info["is_subtechnique"],
            "parent_id": info.get("parent_id"),
            "platforms": info.get("platforms", []),
            "kill_chain_phases": info.get("kill_chain_phases", []),
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(valid, f, ensure_ascii=False, indent=2)

    sub_count = sum(1 for v in valid.values() if v["is_subtechnique"])
    print(f"[OK] Saved {len(valid)} techniques to {output_path}")
    print(f"     - Main techniques: {len(valid) - sub_count}")
    print(f"     - Sub-techniques:  {sub_count}")

    return valid


if __name__ == "__main__":
    build_valid_techniques()