import json
from typing import Dict, Any, List
from normalizer import AlertNormalizer


class AlertProcessor:
    def __init__(self):
        self.seen_ids = set()

    def process_and_save(self, input_file: str, output_file: str) -> List[Dict[str, Any]]:
        """Loads alerts, normalizes them, filters duplicates, and writes to disk."""
        with open(input_file, "r", encoding="utf-8") as f:
            raw_alerts = json.load(f)

        if isinstance(raw_alerts, dict):
            raw_alerts = [raw_alerts]

        processed_alerts = []
        duplicates = 0

        for raw in raw_alerts:
            # Delegate standardization to AlertNormalizer
            alert = AlertNormalizer.normalize(raw)
            alert_id = alert["alert_id"]

            # Prevent duplicate registration
            if alert_id in self.seen_ids:
                duplicates += 1
                continue

            self.seen_ids.add(alert_id)
            processed_alerts.append(alert)

        # Save standardized results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_alerts, f, ensure_ascii=False, indent=2)

        print(f"[+] Processed: {len(processed_alerts)} alerts saved ({duplicates} duplicates skipped).")
        print(f"[+] Output written to: {output_file}")

        return processed_alerts