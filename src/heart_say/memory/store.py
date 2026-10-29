"""对话记忆：Redis 优先，本地 JSON 降级；滚动历史 + 轻量事实抽取。

这一层只做“可解释的短期/长期记忆”：
- 近 N 轮原文，保证指代与情绪连续性；
- 从对话中抽取的稳定事实（孩子年龄、年级、反复出现的困扰）；
- 不做用户画像推断，不存储可识别个人身份的信息。
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from ..config import get_settings
from ..safety import redact_pii

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Turn:
    role: str
    content: str
    created_at: str = field(default_factory=_now)


@dataclass
class Session:
    session_id: str
    messages: list[Turn] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "messages": [asdict(turn) for turn in self.messages],
            "facts": self.facts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        return cls(
            session_id=data["session_id"],
            messages=[Turn(**turn) for turn in data.get("messages", [])],
            facts=data.get("facts", []),
            created_at=data.get("created_at", _now()),
            updated_at=data.get("updated_at", _now()),
        )


_CHILD_AGE = re.compile(r"(?:女儿|儿子|孩子|娃|小孩).{0,6}?(\d{1,2})\s*岁")
_USER_AGE = re.compile(r"我(?:今年)?\s*(\d{1,2})\s*岁")
_GRADE = re.compile(
    r"(小学[一二三四五六]年级|初[一二三]|高[一二三]|七|八|九|十|十一|十二)年级?"
)
_CONCERNS = {
    "学业压力": r"(考试|成绩|作业|排名|升学|中考|高考|厌学|不想上学)",
    "手机/网络": r"(手机|游戏|短视频|网络|沉迷|刷视频)",
    "亲子冲突": r"(吵架|冷战|吼|顶嘴|不听|冲突|关系不好)",
    "睡眠": r"(睡不着|失眠|熬夜|睡眠|睡不好)",
    "同伴关系": r"(同学|朋友|孤立|排挤|霸凌|校园|社交)",
    "情绪低落": r"(难过|低落|抑郁|没意思|哭|委屈|孤独)",
    "焦虑": r"(焦虑|紧张|担心|害怕|心慌|压力大)",
    "自伤风险": r"(自杀|自伤|自残|不想活|割腕|吞药)",
    "身材/进食": r"(胖|瘦|身材|体重|催吐|暴食|不吃饭)",
}


def extract_facts(text: str, existing: list[str], full_history: str = "") -> list[str]:
    facts = list(existing)
    text = redact_pii(text)

    if "孩子年龄" not in " ".join(facts):
        match = _CHILD_AGE.search(text)
        if match:
            facts.append(f"孩子 {match.group(1)} 岁")
    if "用户年龄" not in " ".join(facts):
        match = _USER_AGE.search(text)
        if match:
            facts.append(f"用户 {match.group(1)} 岁")
    if "年级" not in " ".join(facts):
        match = _GRADE.search(text)
        if match:
            facts.append(f"年级：{match.group(0)}")
    if "身份：" not in " ".join(facts):
        if re.search(r"(我(的)?(女儿|儿子|孩子|娃)|我是(他|她)(妈妈|爸爸)|当妈|当爸|作为(妈妈|爸爸))", text):
            facts.append("身份：家长")
        elif re.search(r"(我(妈|爸|父母)|我妈|我爸|我(今年)?\s*1?[0-9]\s*岁)", text):
            facts.append("身份：青少年")

    history = f"{full_history}{text}"
    for theme, pattern in _CONCERNS.items():
        fact = f"反复提到：{theme}"
        if fact in facts:
            continue
        if len(re.findall(pattern, history)) >= 2:
            facts.append(fact)

    # 去重并限制长度，避免 prompt 膨胀
    unique: list[str] = []
    for item in facts:
        if item not in unique:
            unique.append(item)
    return unique[-10:]


class SessionStore:
    def __init__(self, max_turns: int | None = None, session_dir: Path | None = None):
        settings = get_settings()
        self.max_turns = max_turns or settings.max_history_turns
        self.dir = session_dir or settings.sessions_path
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: dict[str, Session] = {}
        self._redis = None
        if settings.redis_url:
            try:
                import redis

                self._redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
                self._redis.ping()
            except Exception as exc:
                logger.warning("Redis 不可用，使用本地会话文件：%s", exc)
                self._redis = None

    def _path(self, session_id: str) -> Path:
        return self.dir / f"{session_id}.json"

    def get(self, session_id: str) -> Session:
        if session_id in self._cache:
            return self._cache[session_id]
        session: Session | None = None
        if self._redis is not None:
            try:
                raw = self._redis.get(f"heart_say:session:{session_id}")
                if raw:
                    session = Session.from_dict(json.loads(raw))
            except Exception as exc:
                logger.warning("读取 Redis 会话失败：%s", exc)
        if session is None and self._path(session_id).exists():
            try:
                session = Session.from_dict(json.loads(self._path(session_id).read_text("utf-8")))
            except Exception as exc:
                logger.warning("读取会话文件失败：%s", exc)
        if session is None:
            session = Session(session_id=session_id)
        self._cache[session_id] = session
        return session

    def append_turn(self, session_id: str, user_text: str, assistant_text: str) -> Session:
        with self._lock:
            session = self.get(session_id)
            safe_user = redact_pii(user_text)
            session.messages.append(Turn(role="user", content=safe_user))
            session.messages.append(Turn(role="assistant", content=assistant_text))
            session.messages = session.messages[-self.max_turns * 2 :]
            history = "".join(turn.content for turn in session.messages[:-2])
            session.facts = extract_facts(safe_user, session.facts, history)
            session.updated_at = _now()
            self._cache[session_id] = session
            self._persist(session)
            return session

    def _persist(self, session: Session) -> None:
        payload = json.dumps(session.to_dict(), ensure_ascii=False)
        if self._redis is not None:
            try:
                self._redis.setex(f"heart_say:session:{session.session_id}", 7 * 24 * 3600, payload)
                return
            except Exception as exc:
                logger.warning("写入 Redis 失败，降级到文件：%s", exc)
        try:
            self._path(session.session_id).write_text(payload, encoding="utf-8")
        except OSError as exc:
            logger.warning("写入会话文件失败：%s", exc)

    def history_for_prompt(self, session_id: str) -> tuple[list[dict[str, str]], list[str]]:
        session = self.get(session_id)
        recent = session.messages[-self.max_turns * 2 :]
        return (
            [{"role": turn.role, "content": turn.content} for turn in recent],
            session.facts,
        )


@lru_cache(maxsize=1)
def get_session_store() -> SessionStore:
    return SessionStore()
