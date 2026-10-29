"""LangGraph 编排：情绪识别 → 安全分级 → 检索 → 提示词组装 → 输出守门。

生成节点不放进图里：API 需要逐 token 流式输出，而图负责确定性决策。
图产出 AgentPlan，由 agent.stream()/respond() 执行生成与收尾，两条路径共享
同一份上下文与安全策略。
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, AsyncIterator, TypedDict

from ..config import get_settings
from ..llm import GenerationContext, LLMError, build_generator
from ..memory import SessionStore, get_session_store
from ..rag import get_retriever
from ..rag.retriever import to_citations
from ..safety import DISCLAIMER, assess_risk, crisis_response, detect_audience, guard_output, resources_for
from ..schemas import ChatResponse, Citation, RetrievedDoc, RiskAssessment, SafetyInfo, TraceInfo
from .emotion import build_retrieval_query, detect_emotion, detect_intent
from .prompts import build_messages
from .tools import run_tools

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    session_id: str
    user_message: str
    use_rag: bool
    history: list[dict[str, str]]
    facts: list[str]
    audience: str
    emotion: str
    emotion_intensity: float
    intent: str
    risk: dict[str, Any]
    crisis_text: str
    docs: list[dict[str, Any]]
    messages: list[dict[str, str]]
    tool_notes: list[str]
    tool_calls: list[dict[str, Any]]
    timings: dict[str, float]
    graph: list[str]


@dataclass
class AgentPlan:
    session_id: str
    user_message: str
    ctx: GenerationContext
    messages: list[dict[str, str]]
    risk: RiskAssessment
    safety: SafetyInfo
    citations: list[Citation]
    trace: TraceInfo
    crisis_text: str = ""
    guard_flags: list[str] = field(default_factory=list)
    streamed_text: str = ""
    prepared_at: float = field(default_factory=time.perf_counter)


def _merge(state: AgentState, key: str, value: Any) -> dict[str, Any]:
    return {key: value}


class HeartSayAgent:
    def __init__(self, store: SessionStore | None = None):
        self.settings = get_settings()
        self.store = store or get_session_store()
        self.generator = build_generator()
        self.retriever = None
        self.graph = self._build_graph()

    def ensure_retriever(self):
        if self.retriever is None:
            self.retriever = get_retriever()
        return self.retriever

    def _build_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph

            builder = StateGraph(AgentState)
            builder.add_node("triage", self._triage_node)
            builder.add_node("safety_protocol", self._protocol_node)
            builder.add_node("tools", self._tools_node)
            builder.add_node("retrieve", self._retrieve_node)
            builder.add_node("compose", self._compose_node)
            builder.add_edge(START, "triage")
            builder.add_conditional_edges(
                "triage",
                self._route_after_triage,
                {"protocol": "safety_protocol", "retrieve": "tools"},
            )
            builder.add_edge("safety_protocol", END)
            builder.add_edge("tools", "retrieve")
            builder.add_edge("retrieve", "compose")
            builder.add_edge("compose", END)
            return builder.compile()
        except Exception as exc:
            logger.warning("LangGraph 初始化失败，使用顺序编排：%s", exc)
            return None

    # ---------- nodes ----------

    async def _triage_node(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        message = state["user_message"]
        facts = state.get("facts", [])
        fallback = "unknown"
        if any("身份：家长" in fact for fact in facts):
            fallback = "parent"
        elif any("身份：青少年" in fact or ("用户" in fact and "岁" in fact) for fact in facts):
            fallback = "teen"

        audience = detect_audience(message, fallback)
        emotion, intensity = detect_emotion(message)
        intent = detect_intent(message)
        risk = assess_risk(message, audience)

        # 高风险不是单轮状态：如果最近几轮出现过自伤/虐待/霸凌，后续追问仍按协议回应；
        # 明确表示“已经安全/有人陪/在医院”时才解除。
        safe_now = re.search(r"(我现在安全了|已经安全|有人陪着我|我在医院|已经联系(上)?(老师|家长|医生)|医生在)", message)
        prior: list[RiskAssessment] = []
        prior_text = ""
        if not safe_now:
            for turn in state.get("history", [])[-6:]:
                if turn.get("role") != "user":
                    continue
                prior_risk = assess_risk(turn.get("content", ""), audience)
                if prior_risk.level in ("high", "critical"):
                    prior.append(prior_risk)
                    prior_text = turn.get("content", "")
        if prior:
            order = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
            level = max([risk.level] + [item.level for item in prior], key=lambda value: order[value])
            categories = sorted({category for item in prior + [risk] for category in item.categories})
            risk = RiskAssessment(
                level=level,
                categories=categories,
                action="protocol",
                reason="历史高风险延续，需保持安全协议",
            )

        crisis_text = ""
        if risk.action == "protocol":
            # 跨轮追问时带上上一轮高风险原话，避免丢掉“朋友/孩子/本人”主语
            context_text = f"{message} {prior_text}" if prior_text else message
            crisis_text = crisis_response(context_text, risk, audience)

        timings = dict(state.get("timings", {}))
        timings["triage_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return {
            "audience": audience,
            "emotion": emotion,
            "emotion_intensity": intensity,
            "intent": intent,
            "risk": risk.model_dump(),
            "crisis_text": crisis_text,
            "timings": timings,
            "graph": state.get("graph", []) + ["triage"],
        }

    def _route_after_triage(self, state: AgentState) -> str:
        return "protocol" if state.get("crisis_text") else "retrieve"

    async def _protocol_node(self, state: AgentState) -> dict[str, Any]:
        return {"graph": state.get("graph", []) + ["safety_protocol"]}

    async def _tools_node(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        risk = RiskAssessment(**state.get("risk", {"level": "none"}))
        notes, calls = run_tools(state["user_message"], risk)
        timings = dict(state.get("timings", {}))
        timings["tools_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return {
            "tool_notes": notes,
            "tool_calls": calls,
            "timings": timings,
            "graph": state.get("graph", []) + ["tools"],
        }

    async def _retrieve_node(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        if not state.get("use_rag", True):
            timings = dict(state.get("timings", {}))
            timings["retrieve_ms"] = 0.0
            return {"docs": [], "timings": timings, "graph": state.get("graph", []) + ["retrieve:off"]}

        retriever = self.ensure_retriever()
        query = build_retrieval_query(
            state["user_message"],
            state.get("history", []),
            state.get("emotion", "unknown"),
            state.get("audience", "unknown"),
            state.get("facts", []),
        )
        docs: list[RetrievedDoc] = []
        try:
            docs = await asyncio.to_thread(
                retriever.retrieve,
                query,
                None,
                state.get("audience", "unknown"),
                state.get("risk", {}).get("level", "none"),
            )
        except Exception as exc:
            logger.warning("检索失败，继续无 RAG 回复：%s", exc)

        timings = dict(state.get("timings", {}))
        timings["retrieve_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return {
            "docs": [doc.model_dump() for doc in docs],
            "timings": timings,
            "graph": state.get("graph", []) + ["retrieve"],
        }

    async def _compose_node(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        risk = RiskAssessment(**state.get("risk", {"level": "none"}))
        ctx = GenerationContext(
            user_message=state["user_message"],
            audience=state.get("audience", "unknown"),
            emotion=state.get("emotion", "unknown"),
            emotion_intensity=state.get("emotion_intensity", 0.0),
            risk_level=risk.level,
            risk_categories=risk.categories,
            docs=[RetrievedDoc(**doc) for doc in state.get("docs", [])],
            facts=state.get("facts", []),
            history=state.get("history", []),
            use_rag=state.get("use_rag", True),
            tool_notes=state.get("tool_notes", []),
        )
        messages = build_messages(ctx)
        timings = dict(state.get("timings", {}))
        timings["compose_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return {
            "messages": messages,
            "timings": timings,
            "graph": state.get("graph", []) + ["compose"],
        }

    # ---------- pipeline ----------

    async def _run_graph(self, initial: AgentState) -> AgentState:
        if self.graph is not None:
            try:
                return await self.graph.ainvoke(initial)
            except Exception as exc:
                logger.warning("LangGraph 执行失败，回退顺序编排：%s", exc)
        state: AgentState = dict(initial)
        state.update(await self._triage_node(state))
        if self._route_after_triage(state) == "protocol":
            state.update(await self._protocol_node(state))
            return state
        state.update(await self._tools_node(state))
        state.update(await self._retrieve_node(state))
        state.update(await self._compose_node(state))
        state["graph"] = state.get("graph", []) + ["sequential"]
        return state

    async def prepare(
        self,
        session_id: str,
        message: str,
        use_rag: bool = True,
        stream_delay_ms: int = 0,
        persist_memory: bool = True,
    ) -> AgentPlan:
        started = time.perf_counter()
        history, facts = self.store.history_for_prompt(session_id)
        initial: AgentState = {
            "session_id": session_id,
            "user_message": message.strip(),
            "use_rag": use_rag,
            "history": history,
            "facts": facts,
            "timings": {},
            "graph": [],
        }
        state = await self._run_graph(initial)

        risk = RiskAssessment(**state.get("risk", {"level": "none"}))
        ctx = GenerationContext(
            user_message=message.strip(),
            audience=state.get("audience", "unknown"),
            emotion=state.get("emotion", "unknown"),
            emotion_intensity=state.get("emotion_intensity", 0.0),
            risk_level=risk.level,
            risk_categories=risk.categories,
            docs=[RetrievedDoc(**doc) for doc in state.get("docs", [])],
            facts=facts,
            history=history,
            use_rag=use_rag,
            stream_delay_ms=0 if state.get("crisis_text") else stream_delay_ms,
            tool_notes=state.get("tool_notes", []),
        )
        messages = state.get("messages", [])
        docs = ctx.docs if use_rag else []
        citations = to_citations(docs) if docs else []

        timings = dict(state.get("timings", {}))
        timings["prepare_ms"] = round((time.perf_counter() - started) * 1000, 2)
        trace = TraceInfo(
            request_id=uuid.uuid4().hex[:12],
            model=self.settings.llm_model if self.settings.llm_backend != "offline" else "offline-composer-v1",
            backend=self.settings.llm_backend,
            use_rag=use_rag,
            audience=state.get("audience", "unknown"),  # type: ignore[arg-type]
            emotion=state.get("emotion", "unknown"),
            intent="crisis" if state.get("crisis_text") else state.get("intent", "support"),
            timings_ms=timings,
            retrieved=[
                {
                    "id": doc.id,
                    "title": doc.title,
                    "score": doc.score,
                    "rerank_score": doc.rerank_score,
                }
                for doc in docs
            ],
            tool_calls=state.get("tool_calls", []),
            graph=state.get("graph", []),
        )
        safety = SafetyInfo(
            level=risk.level,
            categories=risk.categories,
            action="protocol" if state.get("crisis_text") else risk.action,
            disclaimer=DISCLAIMER,
            resources=resources_for(risk.level),
            guard_flags=[],
        )
        return AgentPlan(
            session_id=session_id,
            user_message=message,
            ctx=ctx,
            messages=messages,
            risk=risk,
            safety=safety,
            citations=citations,
            trace=trace,
            crisis_text=state.get("crisis_text", ""),
            prepared_at=started,
        )

    async def stream(self, plan: AgentPlan) -> AsyncIterator[str]:
        """按句守门后流式输出，避免越界内容先到达用户。"""
        if plan.crisis_text:
            parts = re.findall(r"[^。！？!?；;]+[。！？!?；;]?", plan.crisis_text)
            for part in parts:
                if not part.strip():
                    continue
                yield part
                if plan.ctx.stream_delay_ms > 0:
                    await asyncio.sleep(plan.ctx.stream_delay_ms / 1000)
            plan.streamed_text = plan.crisis_text
            return

        buffer = ""
        flags: set[str] = set()
        rendered: list[str] = []
        try:
            async for delta in self.generator.astream(plan.messages, plan.ctx):
                buffer += delta
                while True:
                    match = re.search(r"[。！？!?；;\n]", buffer)
                    if not match:
                        break
                    sentence = buffer[: match.end()]
                    buffer = buffer[match.end() :]
                    clean, hit = guard_output(sentence, "none")
                    flags.update(hit)
                    if clean.strip():
                        rendered.append(clean)
                        yield clean
            if buffer.strip():
                clean, hit = guard_output(buffer, "none")
                flags.update(hit)
                if clean.strip():
                    rendered.append(clean)
                    yield clean
        except LLMError as exc:
            logger.error("生成失败：%s", exc)
            fallback = "抱歉，我这边暂时没有连上回复服务。你可以稍后重试；如果此刻有危险，请立即拨打 120/110 或 12356。"
            rendered.append(fallback)
            yield fallback

        plan.streamed_text = "".join(rendered)
        plan.guard_flags = sorted(flags)

    async def respond(
        self,
        session_id: str,
        message: str,
        use_rag: bool = True,
        persist_memory: bool = True,
    ) -> ChatResponse:
        plan = await self.prepare(session_id, message, use_rag=use_rag, persist_memory=persist_memory)
        if plan.crisis_text:
            text = plan.crisis_text
        else:
            started = time.perf_counter()
            raw = await self.generator.agenerate(plan.messages, plan.ctx)
            text, flags = guard_output(raw, plan.risk.level)
            plan.guard_flags = flags
            plan.trace.timings_ms["generate_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return self.finalize(plan, text)

    def finalize(self, plan: AgentPlan, text: str, persist_memory: bool = True) -> ChatResponse:
        if plan.streamed_text:
            text = plan.streamed_text
            # 流式路径已逐句守门，这里只补全完整文本的标记
            _, final_flags = guard_output(text, "none")
            plan.guard_flags = sorted(set(plan.guard_flags) | set(final_flags))

        plan.safety.guard_flags = plan.guard_flags
        plan.trace.timings_ms["total_ms"] = round((time.perf_counter() - plan.prepared_at) * 1000, 2)

        if persist_memory and text:
            try:
                self.store.append_turn(plan.session_id, plan.user_message, text)
            except Exception as exc:
                logger.warning("写入记忆失败：%s", exc)

        return ChatResponse(
            session_id=plan.session_id,
            message_id=uuid.uuid4().hex[:12],
            text=text,
            citations=plan.citations,
            safety=plan.safety,
            trace=plan.trace,
        )


@lru_cache(maxsize=1)
def get_agent() -> HeartSayAgent:
    return HeartSayAgent()
