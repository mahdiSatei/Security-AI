"""Grounded alert analysis with a language model (section 4 of the spec).

The model never decides which technique an alert maps to: candidates come from the
retrieval stage, and anything outside that allowlist, or unsupported by quoted
evidence, is dropped before the result is returned.
"""

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from pydantic import ValidationError

from llm_client import LLMError, OpenAICompatibleClient
from llm_prompt import build_messages, build_retry_messages
from result_schema import (
    AlertAnalysis,
    MitreTechnique,
    RelatedAlert,
    derive_risk_level,
)

MAX_ATTEMPTS = 2
GROUNDING_THRESHOLD = 0.5
_CODE_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TOKEN = re.compile(r"[a-z0-9][a-z0-9_.\-/]{2,}")


def _prepare_alerts(alerts: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    seen = set()
    for alert in alerts or []:
        alert_id = str(alert.get("alert_id") or "").strip()
        if not alert_id or alert_id in seen:
            continue
        seen.add(alert_id)
        prepared.append(alert)
    return prepared


def _prepare_candidates(
    candidates: Optional[Sequence[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    seen = set()
    for candidate in candidates or []:
        attack_id = str(candidate.get("attack_id") or "").strip().upper()
        if not attack_id or attack_id in seen:
            continue
        seen.add(attack_id)
        prepared.append(candidate)
    return prepared


def _parse_json(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    fenced = _CODE_FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("response did not contain a JSON object")
        text = text[start : end + 1]
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("response JSON was not an object")
    return payload


def _corpus(
    alerts: Sequence[Dict[str, Any]], candidates: Sequence[Dict[str, Any]]
) -> str:
    parts: List[str] = []
    for item in list(alerts) + list(candidates):
        for value in item.values():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, (list, tuple)):
                parts.extend(str(v) for v in value)
    return " ".join(parts).lower()


def _grounded(text: str, corpus: str) -> bool:
    """True when most words of the quoted evidence actually appear in the evidence."""
    tokens = set(_TOKEN.findall(str(text).lower()))
    if not tokens:
        return False
    hits = sum(1 for token in tokens if token in corpus)
    return hits / len(tokens) >= GROUNDING_THRESHOLD


def _short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        errors = exc.errors()
        if errors:
            location = ".".join(str(part) for part in errors[0].get("loc", ()))
            message = errors[0].get("msg", "invalid value")
            return f"{location}: {message}"[:300]
    return str(exc)[:300]


class AlertAnalyzer:
    def __init__(self, client=None, max_attempts: int = MAX_ATTEMPTS):
        self.client = client if client is not None else OpenAICompatibleClient()
        self.max_attempts = max_attempts

    def analyze(
        self,
        question: str,
        alerts: Optional[Sequence[Dict[str, Any]]],
        technique_candidates: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Return the section-5 result dict for a question over retrieved alerts."""
        alert_views = _prepare_alerts(alerts)
        candidates = _prepare_candidates(technique_candidates)

        if not alert_views:
            return self._abstain(
                "No alerts were retrieved, so there is no evidence to analyse.",
                ["abstained: empty retrieval"],
            )

        messages = build_messages(question, alert_views, candidates)
        warnings: List[str] = []
        analysis: Optional[AlertAnalysis] = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                raw = self.client.complete(messages)
            except LLMError as exc:
                warnings.append(f"llm_unavailable: {exc}")
                break
            try:
                analysis = AlertAnalysis.model_validate(_parse_json(raw))
                break
            except (ValueError, ValidationError) as exc:
                error = _short_error(exc)
                warnings.append(f"invalid_output_attempt_{attempt}: {error}")
                messages = build_retry_messages(messages, raw, error)

        if analysis is None:
            return self._fallback(alert_views, warnings)

        return self._finalize(analysis, alert_views, candidates, warnings)

    def _finalize(
        self,
        analysis: AlertAnalysis,
        alerts: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        warnings: List[str],
    ) -> Dict[str, Any]:
        corpus = _corpus(alerts, candidates)
        by_id = {str(alert.get("alert_id")): alert for alert in alerts}
        allowed = {
            str(candidate.get("attack_id")).strip().upper(): candidate
            for candidate in candidates
        }

        related: List[RelatedAlert] = []
        for item in analysis.related_alerts:
            alert_id = item.alert_id.strip()
            source = by_id.get(alert_id)
            if source is None:
                warnings.append(f"dropped_unknown_related_alert:{alert_id}")
                continue
            raw_score = source.get("similarity_score")
            score = float(raw_score) if isinstance(raw_score, (int, float)) else item.similarity_score
            related.append(RelatedAlert(alert_id=alert_id, similarity_score=score))

        techniques: List[MitreTechnique] = []
        for technique in analysis.mitre_techniques:
            technique_id = technique.technique_id.strip().upper()
            candidate = allowed.get(technique_id)
            if candidate is None:
                warnings.append(f"dropped_unverified_technique:{technique_id}")
                continue
            evidence = [item for item in technique.evidence if _grounded(item, corpus)]
            if not evidence:
                warnings.append(f"dropped_unsupported_technique:{technique_id}")
                continue
            techniques.append(
                MitreTechnique(
                    technique_id=technique_id,
                    technique_name=str(
                        candidate.get("name") or technique.technique_name
                    ),
                    evidence=evidence,
                    explanation=technique.explanation.strip(),
                )
            )

        alert_id: Optional[str] = None
        requested_id = (analysis.alert_id or "").strip()
        if requested_id in by_id:
            alert_id = requested_id
        elif len(alerts) == 1:
            alert_id = str(alerts[0].get("alert_id"))

        result = AlertAnalysis(
            alert_id=alert_id,
            summary=analysis.summary.strip(),
            related_alerts=related,
            mitre_techniques=techniques,
            risk_level=derive_risk_level(alerts),
            needs_further_investigation=(
                analysis.needs_further_investigation or not techniques or bool(warnings)
            ),
            warnings=warnings,
        )
        return result.to_dict()

    def _fallback(self, alerts: List[Dict[str, Any]], warnings: List[str]) -> Dict[str, Any]:
        """Deterministic answer used when the model fails (spec challenge 7)."""
        top = alerts[0]
        label = str(top.get("rule_name") or top.get("alert_id"))
        related = [
            RelatedAlert(
                alert_id=str(alert.get("alert_id")),
                similarity_score=alert.get("similarity_score") or 0.0,
            )
            for alert in alerts[1:4]
        ]
        result = AlertAnalysis(
            alert_id=str(top.get("alert_id")) if len(alerts) == 1 else None,
            summary=f"Automated analysis unavailable; retrieved alert {top.get('alert_id')} ({label}).",
            related_alerts=related,
            mitre_techniques=[],
            risk_level=derive_risk_level(alerts),
            needs_further_investigation=True,
            warnings=warnings + ["llm_fallback: deterministic summary only"],
        )
        return result.to_dict()

    @staticmethod
    def _abstain(reason: str, warnings: List[str]) -> Dict[str, Any]:
        return AlertAnalysis(
            summary=reason,
            risk_level=derive_risk_level([]),
            needs_further_investigation=True,
            warnings=warnings,
        ).to_dict()
