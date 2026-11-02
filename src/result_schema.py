"""Structured analysis result for section 5 of the project specification."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class RelatedAlert(BaseModel):
    alert_id: str
    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("similarity_score", mode="before")
    def normalize_score(cls, value):
        # The model sometimes answers with a percentage (84 instead of 0.84).
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        if score > 1.0 and score <= 100.0:
            score = score / 100.0
        return max(0.0, min(1.0, score))


class MitreTechnique(BaseModel):
    technique_id: str
    technique_name: str
    evidence: List[str] = Field(default_factory=list)
    explanation: str = ""


class AlertAnalysis(BaseModel):
    alert_id: Optional[str] = None
    summary: str
    related_alerts: List[RelatedAlert] = Field(default_factory=list)
    mitre_techniques: List[MitreTechnique] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.UNKNOWN
    needs_further_investigation: bool = False
    warnings: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")


_SEVERITY_ORDER = ["low", "medium", "high", "critical"]
_SEVERITY_TO_RISK = {
    "low": RiskLevel.LOW,
    "medium": RiskLevel.MEDIUM,
    "high": RiskLevel.HIGH,
    "critical": RiskLevel.CRITICAL,
}


def derive_risk_level(alerts: List[Dict[str, Any]]) -> RiskLevel:
    """Risk is taken from the alert severities, not from the language model."""
    best = -1
    for alert in alerts:
        severity = str(alert.get("severity") or "").lower()
        if severity in _SEVERITY_ORDER:
            best = max(best, _SEVERITY_ORDER.index(severity))
    if best < 0:
        return RiskLevel.UNKNOWN
    return _SEVERITY_TO_RISK[_SEVERITY_ORDER[best]]
