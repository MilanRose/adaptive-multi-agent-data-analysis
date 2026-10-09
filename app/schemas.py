from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    description: str
    method: Optional[str] = None
    value: Optional[Any] = None


class ModelResult(BaseModel):
    name: str
    metrics: Dict[str, float] = Field(default_factory=dict)


class AgentResult(BaseModel):
    agent: str
    dataset: str
    task: str

    findings: List[str] = Field(default_factory=list)

    evidence: List[Evidence] = Field(default_factory=list)

    model: Optional[ModelResult] = None

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )