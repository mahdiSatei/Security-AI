from datetime import datetime
from typing import Dict, Any


class AlertNormalizer:
    @staticmethod
    def normalize(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts and standardizes core fields from raw alert data."""
        rule = raw.get("rule", {})
        event = raw.get("event", {})
        evidence = raw.get("evidence", {})

        # 1. Alert ID
        alert_id = str(rule.get("id") or raw.get("alert_id") or "UNKNOWN_ID")

        # 2. Timestamp standardization
        timestamp = (
            event.get("@timestamp")
            or raw.get("timestamp")
            or datetime.utcnow().isoformat()
        )

        # 3. Core field extraction with fallbacks
        rule_name = rule.get("name") or raw.get("description", "Unknown Alert")
        severity = str(rule.get("severity") or raw.get("severity", "unknown")).lower()
        source_ip = event.get("source.ip") or raw.get("source_ip")
        destination_ip = event.get("destination.ip") or raw.get("destination_ip")
        target_resource = (
            event.get("url.path")
            or event.get("file.name")
            or raw.get("process_name")
        )
        description = rule.get("logic") or raw.get("description", "")
        evidence_reason = evidence.get("reason")
        matched_pattern = evidence.get("matched_pattern")

        # 4. Dense semantic text representation for Vector DB (Part 2)
        embedding_parts = [
            f"Alert: {rule_name}",
            f"Severity: {severity.upper()}",
            f"Description: {description}",
        ]
        if source_ip:
            embedding_parts.append(f"Source: {source_ip}")
        if target_resource:
            embedding_parts.append(f"Target: {target_resource}")
        if evidence_reason:
            embedding_parts.append(f"Evidence: {evidence_reason}")

        embedding_text = " | ".join(embedding_parts)

        return {
            "alert_id": alert_id,
            "timestamp": timestamp,
            "rule_name": rule_name,
            "severity": severity,
            "risk_score": rule.get("risk_score"),
            "source_ip": source_ip,
            "destination_ip": destination_ip,
            "target_resource": target_resource,
            "description": description,
            "evidence": {
                "reason": evidence_reason,
                "matched_pattern": matched_pattern,
            },
            "embedding_text": embedding_text,
            "raw_data": raw,  # Keeps original data for audit & MITRE mapping
        }