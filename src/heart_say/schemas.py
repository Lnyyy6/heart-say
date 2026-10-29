"""API / Agent 共享的数据结构。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["none", "low", "medium", "high", "critical"]
Audience = Literal["parent", "teen", "unknown"]


class Citation(BaseModel):
    index: int
    doc_id: str
    title: str
    topic: str
    source_name: str
    source_url: str
    snippet: str
    score: float
    rerank_score: float | None = None


class RiskAssessment(BaseModel):
    level: RiskLevel = "none"
    categories: list[str] = Field(default_factory=list)
    action: Literal["none", "support", "protocol"] = "none"
    reason: str = ""


class SafetyInfo(BaseModel):
    level: RiskLevel
    categories: list[str]
    action: str
    disclaimer: str
    resources: list[dict[str, str]] = Field(default_factory=list)
    guard_flags: list[str] = Field(default_factory=list)


class RetrievedDoc(BaseModel):
    id: str
    title: str
    topic: str
    audience: str
    risk_level: str
    source_name: str
    source_url: str
    text: str
    key_points: list[str] = Field(default_factory=list)
    score: float = 0.0
    rerank_score: float | None = None


class TraceInfo(BaseModel):
    request_id: str
    model: str
    backend: str
    use_rag: bool
    audience: Audience = "unknown"
    emotion: str = "unknown"
    intent: str = "support"
    timings_ms: dict[str, float] = Field(default_factory=dict)
    retrieved: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    graph: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=80, pattern=r"^[A-Za-z0-9_-]*$")
    stream: bool = True
    use_rag: bool = True


class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    safety: SafetyInfo
    trace: TraceInfo


class MessageOut(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: str


class SessionOut(BaseModel):
    session_id: str
    messages: list[MessageOut]
    facts: list[str] = Field(default_factory=list)


class HealthOut(BaseModel):
    status: str
    app: str
    version: str
    llm_backend: str
    embedding_backend: str
    kb_docs: int
    qdrant_mode: str
    timestamp: str


class EvalCase(BaseModel):
    id: str
    category: str
    audience: Literal["parent", "teen"]
    turns: list[str]
    expected_risk: RiskLevel
    must_include: list[str] = Field(default_factory=list)
    must_not: list[str] = Field(default_factory=list)
    reference: str = ""
    notes: str = ""
