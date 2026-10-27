from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class StandardizedAlert(BaseModel):
    
    alert_id: str
    timestamp: datetime
    rule_name: str
    severity: SeverityLevel = SeverityLevel.UNKNOWN
    risk_score: Optional[int] = Field(default=None, ge=0, le=100)
    
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    hostname: Optional[str] = None
    username: Optional[str] = None
    
    event_categories: List[str] = Field(default_factory=list)
    action_type: Optional[str] = None 
    target_resource: Optional[str] = None  
    
    description: str
    evidence_reason: Optional[str] = None
    matched_pattern: Optional[str] = None
    
    raw_data: Dict[str, Any]

    @field_validator("severity", mode="before")
    def normalize_severity(cls, v):
        if isinstance(v, str):
            v_lower = v.lower().strip()
            if v_lower in SeverityLevel._value2member_map_:
                return v_lower
        return SeverityLevel.UNKNOWN

    def to_embedding_text(self) -> str:
       
        parts = [
            f"Alert: {self.rule_name}",
            f"Severity: {self.severity.value.upper()}",
            f"Description: {self.description}",
        ]
        if self.source_ip:
            parts.append(f"Source IP: {self.source_ip}")
        if self.destination_ip:
            parts.append(f"Destination IP: {self.destination_ip}")
        if self.target_resource:
            parts.append(f"Target Resource: {self.target_resource}")
        if self.evidence_reason:
            parts.append(f"Evidence: {self.evidence_reason}")
        if self.matched_pattern:
            parts.append(f"Matched Pattern: {self.matched_pattern}")

        return " | ".join(parts)