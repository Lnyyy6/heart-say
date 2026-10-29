"""Agent 工具层：只做确定性、可解释的查询，不让模型自由编造资源。

工具调用结果会进入 prompt（远程模型）与 trace（前端/评测），
离线应答器仍走同一套边界模板，避免“工具结果”变成第二套安全逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..safety import resources_for
from ..schemas import RiskAssessment


@dataclass
class ToolResult:
    name: str
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_trace(self) -> dict[str, Any]:
        return {"name": self.name, "output": self.output, **self.metadata}


def resource_lookup(message: str) -> ToolResult | None:
    if not any(word in message for word in ("热线", "求助电话", "心理援助", "怎么求助", "哪里求助")):
        return None
    resources = resources_for("high")
    contacts = "；".join(f"{item['name']} {item['contact']}" for item in resources)
    return ToolResult(
        name="resource_lookup",
        output=f"可确认的资源：{contacts}。地区资源与服务时间需向当地卫健委/学校核实。",
        metadata={"contacts": [item["contact"] for item in resources]},
    )


def professional_referral(risk: RiskAssessment, message: str) -> ToolResult | None:
    if risk.level not in ("medium", "high") and not any(
        word in message for word in ("诊断", "药", "评估", "去医院", "看医生")
    ):
        return None
    categories = set(risk.categories)
    if risk.level in ("high", "critical"):
        return None  # 危机路径使用固定协议，工具不参与
    if categories & {"medication", "diagnosis", "eating"} or any(
        word in message for word in ("诊断", "药", "评估")
    ):
        target = "孩子" if any(word in message for word in ("孩子", "女儿", "儿子")) else "你"
        return ToolResult(
            name="professional_referral",
            output=(
                f"建议路径：学校心理老师 → 正规医院精神科/心理科；涉及用药时由医生面诊调整。"
                f"不要替{target}下诊断，也不要自行加减药。"
            ),
            metadata={"categories": sorted(categories)},
        )
    return None


def run_tools(message: str, risk: RiskAssessment) -> tuple[list[str], list[dict[str, Any]]]:
    results: list[ToolResult] = []
    resource = resource_lookup(message)
    if resource:
        results.append(resource)
    referral = professional_referral(risk, message)
    if referral:
        results.append(referral)
    notes = [result.output for result in results]
    calls = [result.as_trace() for result in results]
    return notes, calls
