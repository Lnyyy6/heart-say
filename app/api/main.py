"""心语 FastAPI 服务：流式聊天、会话记忆、安全元信息、评测看板。"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from heart_say import __version__
from heart_say.agent import get_agent
from heart_say.config import get_settings
from heart_say.memory import get_session_store
from heart_say.rag.store import read_manifest
from heart_say.safety import DISCLAIMER, resources_for
from heart_say.schemas import (
    ChatRequest,
    ChatResponse,
    HealthOut,
    MessageOut,
    SessionOut,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("heart_say.api")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        agent = get_agent()
        agent.ensure_retriever()  # 预热索引/模型
        logger.info(
            "服务启动：backend=%s embedding=%s reranker=%s",
            settings.llm_backend,
            settings.embedding_backend,
            settings.reranker_backend,
        )
    except Exception as exc:
        logger.warning("预热失败，将在首个请求时重试：%s", exc)
    yield


app = FastAPI(
    title="心语 · 心理教育情绪陪伴助手",
    version=__version__,
    description="多轮共情对话、RAG 引用、危机识别与转介。非医疗诊断工具。",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


def _health_payload() -> HealthOut:
    manifest = read_manifest()
    chunks = int(manifest.get("chunks", 0))
    mode = str(manifest.get("store_mode", "unknown"))
    try:
        agent = get_agent()
        retriever = agent.ensure_retriever()
        chunks = retriever.store.count()
        mode = retriever.store.mode
    except Exception as exc:
        logger.warning("健康检查读取索引失败：%s", exc)
    return HealthOut(
        status="ok",
        app=settings.app_name,
        version=__version__,
        llm_backend=settings.llm_backend,
        embedding_backend=settings.embedding_backend,
        kb_docs=chunks,
        qdrant_mode=mode,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/", include_in_schema=False)
async def root():
    return {
        "app": settings.app_name,
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
        "ui": "Next.js 前端默认运行在 http://localhost:3000",
    }


@app.get("/health", response_model=HealthOut)
async def health():
    return _health_payload()


@app.get("/meta")
async def meta():
    return {
        "app": settings.app_name,
        "version": __version__,
        "tagline": "不急着给答案，先陪你把感受说清楚。",
        "disclaimer": DISCLAIMER,
        "resources": resources_for("high"),
        "suggestions": [
            "最近孩子总说不想上学，我该怎么开口？",
            "孩子把自己关在房间，不愿意跟我说话。",
            "我一考试就睡不着，总怕考砸。",
            "我好像总是控制不住吼孩子，事后又很自责。",
        ],
        "capabilities": [
            "多轮共情对话与短期记忆",
            "心理教育知识库检索与可追溯引用",
            "危机信号识别与安全转介",
            "家长 / 青少年双视角",
        ],
        "limitations": [
            "不做诊断、不提供药物或剂量建议",
            "不能替代心理咨询、精神科治疗或紧急救援",
            "知识库为公开科普材料改写，不构成医疗建议",
        ],
        "llm_backend": settings.llm_backend,
    }


@app.get("/sessions/{session_id}", response_model=SessionOut)
async def get_session(session_id: str):
    session = get_session_store().get(session_id)
    return SessionOut(
        session_id=session.session_id,
        messages=[
            MessageOut(role=turn.role, content=turn.content, created_at=turn.created_at)
            for turn in session.messages
            if turn.role in ("user", "assistant")
        ],
        facts=session.facts,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """非流式接口，便于评测与调试。"""
    session_id = request.session_id or uuid.uuid4().hex[:12]
    agent = get_agent()
    try:
        return await agent.respond(
            session_id=session_id,
            message=request.message.strip(),
            use_rag=request.use_rag,
        )
    except Exception as exc:
        logger.exception("chat failed")
        raise HTTPException(status_code=502, detail=f"生成失败：{exc}") from exc


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, http_request: Request):
    """SSE 流式接口。事件：meta / delta / citations / done / error。"""
    session_id = request.session_id or uuid.uuid4().hex[:12]
    agent = get_agent()

    async def event_generator():
        started = time.perf_counter()
        try:
            plan = await agent.prepare(
                session_id=session_id,
                message=request.message.strip(),
                use_rag=request.use_rag,
                stream_delay_ms=14,
            )
            yield {
                "event": "meta",
                "data": json.dumps(
                    {
                        "session_id": session_id,
                        "request_id": plan.trace.request_id,
                        "safety": plan.safety.model_dump(),
                        "trace": plan.trace.model_dump(),
                    },
                    ensure_ascii=False,
                ),
            }

            parts: list[str] = []
            async for chunk in agent.stream(plan):
                if await http_request.is_disconnected():
                    logger.info("客户端断开，停止生成 session=%s", session_id)
                    return
                parts.append(chunk)
                yield {"event": "delta", "data": json.dumps({"text": chunk}, ensure_ascii=False)}

            response = agent.finalize(plan, "".join(parts))
            if plan.citations:
                yield {
                    "event": "citations",
                    "data": json.dumps(
                        [citation.model_dump() for citation in plan.citations], ensure_ascii=False
                    ),
                }
            yield {
                "event": "done",
                "data": json.dumps(response.model_dump(), ensure_ascii=False),
            }
            logger.info(
                "chat done session=%s risk=%s elapsed=%.0fms",
                session_id,
                response.safety.level,
                (time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            logger.exception("stream failed")
            yield {
                "event": "error",
                "data": json.dumps(
                    {"message": f"生成失败，请稍后重试：{exc}"}, ensure_ascii=False
                ),
            }

    return EventSourceResponse(event_generator(), ping=15000)


def _load_report(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@app.get("/eval/summary")
async def eval_summary():
    """读取离线评测产物，供前端看板展示。"""
    report_dir = settings.report_dir
    summary = _load_report(report_dir / "summary.json") if report_dir.exists() else None
    runs = []
    if report_dir.exists():
        for path in sorted(report_dir.glob("eval_*.json")):
            report = _load_report(path)
            if report:
                runs.append(
                    {
                        "file": path.name,
                        "name": report.get("name", path.stem),
                        "generated_at": report.get("generated_at"),
                        "cases": report.get("cases", 0),
                        "use_rag": report.get("use_rag"),
                        "model": report.get("model"),
                        "overall": report.get("overall"),
                        "pass_rate": report.get("pass_rate"),
                    }
                )
    if not summary and not runs:
        return JSONResponse(
            status_code=200,
            content={
                "available": False,
                "message": "尚未生成评测报告，运行 `make eval` 后刷新。",
            },
        )
    return {"available": True, "summary": summary, "runs": runs}


@app.get("/eval/report/{name}")
async def eval_report(name: str):
    if "/" in name or ".." in name:
        raise HTTPException(status_code=400, detail="非法文件名")
    path = settings.report_dir / f"{name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="报告不存在")
    return _load_report(path)
