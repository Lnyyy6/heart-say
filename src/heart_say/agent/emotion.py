"""情绪、意图、受众与检索 query 的轻量识别。

规则识别服务于可解释路由；换成模型分类器时保持函数签名即可。
"""

from __future__ import annotations

import re

_EMOTION_LEXICON: dict[str, list[str]] = {
    "anxiety": ["焦虑", "紧张", "担心", "害怕", "心慌", "压力", "睡不着", "不安", "恐惧", "考砸"],
    "sadness": ["难过", "伤心", "哭", "失落", "委屈", "孤独", "没意思", "低落", "想哭", "空落落"],
    "anger": ["生气", "愤怒", "烦", "气死", "吵架", "吼", "不公平", "讨厌", "火大", "顶嘴"],
    "shame": ["自责", "内疚", "羞耻", "没用", "失败", "都是我的错", "丢人", "不够好", "后悔"],
    "fatigue": ["累", "撑不住", "崩溃", "耗尽", "疲惫", "不想动", "没力气", "熬不住"],
    "confusion": ["不知道", "怎么办", "纠结", "迷茫", "乱", "拿不准", "该不该"],
    "joy": ["开心", "进步", "欣慰", "轻松", "好转", "高兴", "有改善"],
}

_INTENSITY = re.compile(r"(很|非常|特别|太|真的|一直|每天|整夜|快|已经|完全|受不了|撑不住)")
_INTENT_PATTERNS = {
    "greeting": re.compile(r"^(你好|您好|hi|hello|在吗|嗨)[!！。~～\s]*$", re.IGNORECASE),
    "boundary": re.compile(r"(是不是|算不算|确诊|诊断|吃什么药|剂量|开点药|保密|你是医生|保证)"),
    "info": re.compile(r"(为什么|怎么办|正常吗|该不该|如何|怎么|什么是|能不能|有没有用)"),
    "action": re.compile(r"(怎么跟|怎么和|如何沟通|怎么开口|怎么引导|怎么处理|有什么办法)"),
    "vent": re.compile(r"(真的|太|受不了|气死|崩溃|想哭|没人懂|委屈)"),
}


def detect_emotion(text: str) -> tuple[str, float]:
    text = re.sub(r"\s+", "", text or "")
    scores: dict[str, int] = {}
    for emotion, words in _EMOTION_LEXICON.items():
        score = sum(text.count(word) for word in words)
        if score:
            scores[emotion] = score
    if not scores:
        return "unknown", 0.0
    emotion = max(scores, key=lambda key: scores[key])
    intensity = min(1.0, 0.35 + 0.2 * scores[emotion] + (0.15 if _INTENSITY.search(text) else 0))
    return emotion, round(intensity, 2)


def detect_intent(text: str) -> str:
    text = (text or "").strip()
    for intent, pattern in _INTENT_PATTERNS.items():
        if pattern.search(text):
            return intent
    return "support"


_EMOTION_QUERY = {
    "anxiety": "焦虑 压力",
    "sadness": "难过 低落 陪伴",
    "anger": "生气 冲突 沟通",
    "shame": "自责 自我接纳",
    "fatigue": "疲惫 休息 支持",
    "confusion": "怎么办 选择",
    "joy": "进步 鼓励",
    "unknown": "情绪 支持",
}

_ANAPHORA = re.compile(r"^(那|这|他|她|它|还是|然后|所以|可是|但是|如果|怎么办|真的)")
_DECISION = re.compile(r"(该不该|要不要|能不能|是不是该|该怎么办|那怎么办)")


def build_retrieval_query(
    message: str,
    history: list[dict[str, str]],
    emotion: str,
    audience: str,
    facts: list[str] | None = None,
) -> str:
    message = (message or "").strip()
    previous_user = ""
    for turn in reversed(history):
        if turn.get("role") == "user":
            previous_user = turn.get("content", "")
            break

    parts = [message]
    if previous_user and (len(message) < 18 or _ANAPHORA.search(message) or _DECISION.search(message)):
        parts.insert(0, previous_user[-160:])

    # 只在原话很短、或没有具体主题词时补情绪/记忆词，避免“焦虑 压力”把检索带偏
    has_topic = re.search(
        r"(孩子|女儿|儿子|上学|厌学|考试|中考|高考|手机|游戏|睡|同学|朋友|老师|作业|成绩|"
        r"吵架|冷战|房间|说话|吃饭|体重|身材|自伤|自杀|离婚|二胎)",
        message,
    )
    if len(message) < 12 or not has_topic:
        theme = _EMOTION_QUERY.get(emotion, "")
        if theme:
            parts.append(theme)
        for fact in (facts or []):
            if fact.startswith("反复提到："):
                parts.append(fact.replace("反复提到：", ""))

    # 去重，限制 query 长度
    seen: set[str] = set()
    tokens = []
    for part in parts:
        for token in re.split(r"[\s，。；;、]+", part):
            if token and token not in seen:
                tokens.append(token)
                seen.add(token)
    return " ".join(tokens)[:400]
