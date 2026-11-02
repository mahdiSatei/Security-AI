"""End-to-end demo: question -> retrieved alerts -> MITRE candidates -> analysis.

Usage (from the project root, or from src/):
    python src/run_analysis.py "which alerts look like a PowerShell attack?"
    python src/run_analysis.py "compare WEB-1001 with similar alerts" --out result.json
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from llm_analyzer import AlertAnalyzer
from llm_client import LLMError
from mitre_matcher import MitreCandidateRetriever
from vector_store import AlertVectorStore

STANDARDIZED_ALERTS = os.path.join("data", "standardized_alerts.json")
DEFAULT_TOP_ALERTS = 5


def load_standardized_alerts(path: str = STANDARDIZED_ALERTS) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        alerts = json.load(handle)
    if isinstance(alerts, dict):
        alerts = [alerts]
    return alerts


def pick_anchor(
    question: str, retrieved: List[Dict[str, Any]], all_alerts: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Prefer an alert the question names explicitly, otherwise the top hit."""
    lowered = question.lower()
    for alert in all_alerts:
        alert_id = str(alert.get("alert_id") or "").lower()
        if alert_id and alert_id in lowered:
            return alert
    return retrieved[0] if retrieved else None


def retrieve_techniques(anchor: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not anchor:
        return []
    try:
        return MitreCandidateRetriever().retrieve_candidates(anchor)
    except Exception as exc:  # index or embedding model missing: degrade, do not crash
        print(f"[!] MITRE candidates unavailable ({exc}); continuing without them.", file=sys.stderr)
        return []


def main(argv: Optional[List[str]] = None, client=None) -> int:
    parser = argparse.ArgumentParser(description="Analyse retrieved security alerts with the LLM.")
    parser.add_argument("question", help="question about the alerts, e.g. 'which alerts show SQL injection?'")
    parser.add_argument("--alerts", default=STANDARDIZED_ALERTS, help="standardized alerts JSON")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_ALERTS, help="retrieved alerts to use")
    parser.add_argument("--out", default=None, help="also write the result JSON to this path")
    args = parser.parse_args(argv)

    all_alerts = load_standardized_alerts(args.alerts)

    try:
        retrieved = AlertVectorStore().search(query=args.question, n_results=args.top)
    except Exception as exc:
        print(f"[!] Alert retrieval failed: {exc}", file=sys.stderr)
        print(
            "    The embedding model is required (pip install sentence-transformers) and the index "
            "must be built first; run the pipeline in Colab or install the dependency locally.",
            file=sys.stderr,
        )
        return 2

    anchor = pick_anchor(args.question, retrieved, all_alerts)
    candidates = retrieve_techniques(anchor)

    print(f"[*] Question: {args.question}")
    print(f"[*] Retrieved alerts: {[a.get('alert_id') for a in retrieved]}")
    print(f"[*] Anchor alert: {anchor.get('alert_id') if anchor else None}")
    print(f"[*] MITRE candidates: {[c.get('attack_id') for c in candidates]}")
    print("-" * 70)

    try:
        result = AlertAnalyzer(client=client).analyze(args.question, retrieved, candidates)
    except LLMError as exc:
        print(f"[!] LLM analysis unavailable: {exc}", file=sys.stderr)
        return 3

    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(payload)
        print(f"[+] Result written to {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
