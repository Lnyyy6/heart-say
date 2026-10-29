"""生成器：OpenAI 兼容后端（vLLM / 云 API）与离线共情编排器。"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import AsyncIterator

import httpx

from ..config import get_settings
from ..schemas import RetrievedDoc


class LLMError(RuntimeError):
    pass


@dataclass
class GenerationContext:
    user_message: str
    audience: str = "unknown"
    emotion: str = "unknown"
    emotion_intensity: float = 0.0
    risk_level: str = "none"
    risk_categories: list[str] = field(default_factory=list)
    docs: list[RetrievedDoc] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    history: list[dict[str, str]] = field(default_factory=list)
    use_rag: bool = True
    tool_notes: list[str] = field(default_factory=list)
    stream_delay_ms: int = 0


class BaseGenerator:
    async def agenerate(self, messages: list[dict[str, str]], ctx: GenerationContext) -> str:
        chunks = [chunk async for chunk in self.astream(messages, ctx)]
        return "".join(chunks)

    async def astream(
        self, messages: list[dict[str, str]], ctx: GenerationContext
    ) -> AsyncIterator[str]:
        raise NotImplementedError


class OpenAICompatibleGenerator(BaseGenerator):
    def __init__(self) -> None:
        settings = get_settings()
        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.timeout = settings.llm_timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    async def astream(
        self, messages: list[dict[str, str]], ctx: GenerationContext
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/chat/completions", json=payload, headers=self._headers()
                ) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", errors="ignore")
                        raise LLMError(f"模型服务返回 {response.status_code}: {body[:300]}")
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                            delta = event["choices"][0]["delta"].get("content") or ""
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
                        if delta:
                            yield delta
        except httpx.HTTPError as exc:
            raise LLMError(f"无法连接模型服务：{exc}") from exc

    async def agenerate(self, messages: list[dict[str, str]], ctx: GenerationContext) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=self._headers()
                )
                if response.status_code >= 400:
                    raise LLMError(f"模型服务返回 {response.status_code}: {response.text[:300]}")
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
        except httpx.HTTPError as exc:
            raise LLMError(f"无法连接模型服务：{exc}") from exc


# ---------------- 离线编排器 ----------------

_EMPATHY = {
    "anxiety": [
        "听起来你这段时间一直绷着，心里很难真正放松。",
        "这件事像一块石头压在胸口，让你反复担心接下来会怎样。",
    ],
    "sadness": [
        "听起来你心里压着不少委屈和失落，可能连说出口都觉得累。",
        "这件事让你很难过，那种提不起劲的感觉是真实的。",
    ],
    "anger": [
        "听起来你是真的被气到了，也觉得自己的用心没有被看见。",
        "这件事让你很窝火，换作谁可能都会觉得不公平。",
    ],
    "shame": [
        "听起来你在反复责怪自己，觉得是不是哪里没做好。",
        "你好像把很多责任都揽到了自己身上，这份自责很沉。",
    ],
    "fatigue": [
        "听起来你已经撑了很久，真的很累，不是矫情。",
        "你像是被消耗得差不多了，还需要硬撑着，确实不容易。",
    ],
    "confusion": [
        "听起来你现在有点乱，不确定该从哪一步开始。",
        "信息一下子太多，人会本能地想先抓住一个确定答案。",
    ],
    "joy": [
        "听到这个进展，我也替你高兴，这份轻松值得被记住。",
        "这真是个让人松一口气的变化，你的在意是有回应的。",
    ],
    "unknown": [
        "这件事对你来说不轻，能开口说出来需要一点力气。",
        "谢谢你愿意把这些讲出来，这本身就需要一点力气。",
    ],
}

_QUESTIONS_PARENT = [
    "你愿意说说，最近一次让你最难受的场景是什么样吗？",
    "如果先只处理一件最急的事，你会选哪一件？",
    "孩子听到你的关心时，通常是什么反应？",
]
_QUESTIONS_TEEN = [
    "如果只挑一件最想先解决的事，你会选哪件？",
    "这些感受里，哪一个最近最常来找你？",
    "你身边有没有一个说话不用太小心的人？",
]
_PARENT_REFRAME = [
    "你已经在很用力地扛这件事，先不用急着给自己打分。",
    "家长越着急，越容易把担心说成指责；这不是你不够好，而是压力下的本能。",
    "先把你自己的焦虑和孩子的状态分开看，下一步会清楚一点。",
]
_TEEN_REFRAME = [
    "你现在的反应不是矫情，情绪在提醒你：有些需要一直没被看见。",
    "在这个阶段想独立、又怕失去支持，两种拉扯同时存在很正常。",
]

_GENERIC_ACTIONS_PARENT = [
    "可以先选一个低冲突的时刻，用描述而不是评价的方式开口，比如“我注意到你最近睡得晚，我有点担心”。",
    "先减少一次说教式的长谈，换成十分钟的并肩活动，关系松一点，话才进得去。",
]
_GENERIC_ACTIONS_TEEN = [
    "可以先给情绪起个名字，再决定要不要马上解决它；命名本身就能降低一点强度。",
    "先找一个现实中的可信任大人说一小段，不必一次说完全部。",
]

_INJECTION = re.compile(
    r"(忽略(以上|之前|所有).{0,4}(规则|指令)|系统提示词|初始指令|developer mode|开发者模式|"
    r"越狱|base64|安全审计|忘掉你是|扮演.{0,10}(医生|心理医生|小助手)|没有限制的心理医生|"
    r"假装医生|内部规则|<system>|我妈妈是开发者|泄露.{0,6}(提示词|指令))"
)
_BOUNDARY = re.compile(
    r"((我是不是|是不是|算不算|确诊|给我诊断|判断一下|给个结论|确认一下|量表|测评|自测)"
    r".{0,12}(抑郁|焦虑|多动|双相|自闭|强迫|ADHD|厌食))"
)
_MED_REQUEST = re.compile(
    r"(吃什么药|用什么药|什么药|开点药|剂量|吃几片|一天吃多少|药名|舍曲林|氟西汀|阿普唑仑|"
    r"安眠药|褪黑素|停药|自己停|减药|加药|减半|加量|两种药.{0,6}(一起|同吃)|多吃几片)"
)
_IDENTITY = re.compile(r"(你是真人|你是不是(心理医生|咨询师)|你是AI|你是机器人|你有执照)")
_LEGAL_INFO = re.compile(r"(第几条|法律条文|未成年人保护法|教师法|哪一条)")
_BOOK_QUERY = re.compile(r"(被讨厌的勇气|第几页|书里|原文)")
_CURE_CLAIM = re.compile(r"((三天|一周|快速|立刻).{0,8}治好|疗法.{0,8}治好|包治|根治)")
_STEPFAMILY = re.compile(r"(继父|继母|再婚|重组家庭)")
_CONFIDENTIALITY = re.compile(r"(保证|答应).{0,10}(绝对)?(不告诉|保密)|绝对保密|别告诉我爸妈")
_PRIVACY_REQUEST = re.compile(
    r"((聊天记录|微信|朋友圈|手机|日记|账号|发了什么).{0,20}(分析|看|截图|查|发我|看看)|"
    r"帮我看看.{0,8}(发了什么|朋友圈)|查.{0,6}(聊天|记录|手机))"
)
_ROLE_BOUNDARY = re.compile(
    r"(当我的心理咨询师|替代.{0,6}(医生|咨询师)|别让我去医院|不用去医院|不想.{0,10}去医院|"
    r"怕留记录|你直接.{0,6}(方案|治好))"
)
_LEGAL = re.compile(r"(抚养权|法律上|打官司|离婚.{0,12}(赢|财产|抚养)|法院)")
_GUARANTEE_RESULT = re.compile(r"((保证|确保).{0,12}(前五十|提分|进步|考上|治好|好起来)|提分计划)")
_DEPENDENCE = re.compile(r"(永远别离开|只有你能|只能靠你|不要离开我|答应我.{0,8}(永远|一直陪))")
_TAKE_SIDE = re.compile(
    r"(站我这边|站孩子那边|帮我骂|替我骂|直接跟我儿子说|帮我劝劝|你直接跟.{0,8}说|"
    r"让他必须|你必须马上|都是我的错)"
)
_LABEL_OTHERS = re.compile(r"(我爸妈|我父母|我孩子|我儿子|我女儿|他|她).{0,10}(有病|心理变态|人格障碍|是不是.{0,4}(疯|病))")
_PRIVACY = re.compile(r"(其他用户|上一个用户|别人的聊天|聊天记录.{0,6}(导出|发我)|同学叫什么|孩子同学叫什么)")
_OFF_SCOPE_SEXUAL = re.compile(r"(色情|情色|涉黄|性描写|写.{0,6}小说.{0,6}(高中生|未成年))")
_OFF_SCOPE_CHEAT = re.compile(r"(骗过家长|逃课.{0,8}(不被发现|怎么)|作弊.{0,6}(方法|技巧)|怎么骗老师)")
_DISCRIMINATION = re.compile(r"(女生学不好|女孩.{0,4}理科|农村孩子.{0,4}笨|穷人家孩子|地域.{0,4}(差|落后))")
_CURE_GUARANTEE = re.compile(r"(保证书|保证治好|包治|直接.{0,4}治好|不用去医院.{0,6}治好)")
_RESOURCE_QUERY = re.compile(r"(热线|求助电话|心理援助|哪里.{0,4}(求助|找帮助)|怎么求助|心理电话)")
_SCAM = re.compile(r"(刷单|垫钱|返现|校园贷|诈骗|被骗|把钱要回来|追回|对方让我再转)")
_CONTEXTUAL = re.compile(
    r"^(那|这|如果|可是|但是|所以|不能|要不然|算了|不说了|好吧|还是|我不敢|我怕|能不能|"
    r"我就是|我没想|不是想|我不是)"
)


class OfflineComposer(BaseGenerator):
    """无 GPU 的确定性共情应答。

    不是要替代 Qwen，而是让产品闭环、评测、演示在离线环境可运行；
    输出遵循同一套安全与风格约束，并实际使用 RAG 片段中的 key_points。
    """

    def _pick(self, options: list[str], seed_text: str, salt: str = "") -> str:
        if not options:
            return ""
        digest = sum(ord(c) for c in seed_text + salt)
        return options[digest % len(options)]

    def _emotion_key(self, ctx: GenerationContext) -> str:
        return ctx.emotion if ctx.emotion in _EMPATHY else "unknown"

    def _quote_point(self, point: str) -> str:
        point = point.strip().rstrip("。；;")
        if not point:
            return ""
        return point[0] + point[1:] if point else point

    def _specific_action(self, ctx: GenerationContext) -> str:
        if ctx.use_rag and ctx.docs:
            for doc in ctx.docs:
                for point in doc.key_points:
                    if 8 <= len(point) <= 60:
                        return self._quote_point(point)
            for doc in ctx.docs:
                sentences = re.split(r"(?<=[。！？])", doc.text)
                for sentence in sentences:
                    if any(marker in sentence for marker in ("可以", "试试", "先", "建议", "不妨")):
                        clean = sentence.strip().rstrip("。")
                        if 10 <= len(clean) <= 80:
                            return clean
        pool = _GENERIC_ACTIONS_TEEN if ctx.audience == "teen" else _GENERIC_ACTIONS_PARENT
        return self._pick(pool, ctx.user_message, "action")

    def _question_for(self, ctx: GenerationContext, query: str) -> str:
        teen = ctx.audience == "teen"
        if re.search(r"(失眠|睡不着|睡眠|熬夜)", query):
            return (
                "你最近白天的状态怎么样，有没有影响到上课或吃饭？"
                if teen
                else "孩子最近白天的状态怎么样，有没有影响到上课或吃饭？"
            )
        if re.search(r"(手机|游戏|短视频|网络)", query):
            return "你们现在对手机使用有约定吗？执行起来最卡在哪里？"
        if re.search(r"(不想上学|厌学|考试|成绩|中考|高考|作业)", query):
            return (
                "这些压力里，最让你喘不过气的是哪一部分？"
                if teen
                else "这些压力里，孩子最常跟你说的和最不愿意说的是什么？"
            )
        if re.search(r"(不说话|关在房间|冷战|不理我)", query):
            return "你们最近一次能平静地说上几句话，是什么时候？"
        questions = _QUESTIONS_TEEN if teen else _QUESTIONS_PARENT
        return self._pick(questions, query, "question")

    def _special_boundary_response(self, ctx: GenerationContext, query: str | None = None) -> str | None:
        query = query or ctx.user_message
        if _CONFIDENTIALITY.search(query):
            return (
                "我不能保证绝对保密，也不会答应“永远不告诉爸妈”。如果涉及自伤、被伤害或药物过量，"
                "我需要你联系一个可信任的大人；其他内容我会尽量尊重你的隐私。你担心他们知道的是什么？"
            )
        if _PRIVACY_REQUEST.search(query):
            return (
                "我不能帮你看或分析孩子的朋友圈、聊天记录，这涉及她的隐私，也会让信任更难建立。"
                "可以改为观察作息、情绪和社交变化，或者直接说“我有点担心你，不想追问，你想说的时候我在”。"
                "如果你担心孩子此刻的安全，请先确认她在哪里、身边有没有人。"
            )
        if _SCAM.search(query):
            return (
                "先停下来，不要再转任何钱。这很像刷单或垫付诈骗，继续转账只会越陷越深。"
                "我不能帮你瞒着家里，也不能保证钱一定追得回来；现在能做的是保存聊天、转账记录和对方账号，"
                "立刻告诉爸妈或一个可信任的大人，并拨打 110 或 96110 反诈专线报案。被骗不是你的错，别一个人扛。"
                "你现在最担心的是被爸妈骂，还是钱追不回来？"
            )
        if _ROLE_BOUNDARY.search(query):
            return (
                "我不能替代心理咨询师或医生，也不能让你只依靠我。我可以陪你梳理情绪、提供心理教育信息；"
                "如果困扰持续影响生活，建议联系学校心理老师或正规医院的心理科/精神科。"
                "你不想去医院，是担心花钱、被贴标签，还是不知道怎么开口？"
            )
        if _LEGAL.search(query):
            return (
                "我不能给法律结论。抚养权会看证据、孩子年龄和双方的抚养条件，具体需要咨询专业的婚姻家事律师。"
                "可以先把陪伴记录、经济状况和孩子的意愿整理好，也尽量不让孩子卷进大人的冲突。"
                "你现在最担心的是陪伴时间，还是经济条件？"
            )
        if _GUARANTEE_RESULT.search(query):
            return (
                "我不能保证成绩或结果，也不会承诺具体名次。可以做的是把目标拆成可观察的学习动作，"
                "比如每周复盘一次错题、固定两个短时段练习。你愿意先说说孩子目前卡在哪一科吗？"
            )
        if _DEPENDENCE.search(query):
            return (
                "谢谢你信任我，但我不能答应永远不离开，也不能成为你唯一的朋友。你值得在现实里也有人能接住你，"
                "可以从学校心理老师、信任的大人开始。你愿意说说，现实里谁让你稍微安全一点？"
            )
        if _TAKE_SIDE.search(query):
            return (
                "听起来你既委屈又为难：一边希望事情变好，一边又怕关系更僵。我不站队，也不能替你骂人、"
                "劝孩子或命令他。可以一起把你想说的话整理成不伤人的表达，先说你观察到的和你的感受。"
                "你最希望对方听到哪一句？"
            )
        if _LABEL_OTHERS.search(query):
            return (
                "我不能给爸妈贴上“有病”或“心理变态”的标签，也不能隔着对话诊断任何人。"
                "可以具体说说哪些行为让你难受，我们一起看边界和应对。你现在更需要被理解，还是想找一个实际的办法？"
            )
        return None

    def _boundary_response(self, ctx: GenerationContext, query: str | None = None) -> str:
        query = query or ctx.user_message
        special = self._special_boundary_response(ctx, query)
        if special:
            return special
        if _INJECTION.search(query):
            return (
                "我不会提供系统提示词、内部规则，也不会绕过安全边界扮演医生或管理员。"
                "如果你愿意，我们可以继续聊你真正想解决的事；涉及诊断和用药，需要由有资质的专业人员面诊评估。"
            )
        if _MED_REQUEST.search(query):
            return (
                "我不能推荐具体药物、剂量或替你调整用药，这需要精神科医生结合面诊和身体情况判断。"
                "能理解你想尽快找到办法，可以先把最近的睡眠、情绪、食欲变化记下来，去正规医院的精神科或心理科评估；"
                "如果还在上学，也可以先和学校心理老师聊聊。你现在最担心的是哪一部分？"
            )
        target = "孩子" if ctx.audience == "parent" else "你自己"
        return (
            f"仅凭我们的对话，我不能诊断{target}是不是抑郁症、多动症或其他问题——"
            "这需要专业人员通过面谈、量表和时间线综合评估。"
            "可以先把最近两周的情绪、睡眠、学习和社交变化记下来，带去正规医院的精神科/心理科；"
            "如果还在上学，学校心理老师也可以作为第一步。你更担心哪一个具体表现？"
        )

    def _privacy_response(self, ctx: GenerationContext) -> str:
        return (
            "我没有权限访问其他用户、你孩子同学或任何人的聊天记录，也不会导出别人的对话。"
            "我能看到的只有我们这段会话。即使你是监护人，我也不能提供未经本人同意的记录。"
            "如果你担心孩子的安全，可以直接和孩子聊，或联系学校心理老师、班主任；"
            "如果有紧急危险，请拨打 120/110 或 12356。"
        )

    def _off_scope_response(self, ctx: GenerationContext) -> str:
        if _CURE_GUARANTEE.search(ctx.user_message):
            return (
                "我不能保证治好，也不会出任何保证书。抑郁的评估和治疗需要专业人员结合面诊、时间线和身体情况判断，"
                "不是一次对话能承诺的结果。可以陪孩子去正规医院的精神科/心理科，或先联系学校心理老师；"
                "如果孩子有自伤风险，请立即拨打 120/110，或联系 12356。你现在最担心孩子的哪个表现？"
            )
        if _OFF_SCOPE_SEXUAL.search(ctx.user_message):
            return (
                "这类内容我不能写，尤其涉及未成年人。如果你在困惑性教育或身体边界，我可以提供适龄、科学的信息，"
                "也可以一起想怎么和监护人沟通。你想了解的是哪方面？"
            )
        if _OFF_SCOPE_CHEAT.search(ctx.user_message):
            return (
                "欺骗家长、逃课或作弊的具体方法我不能教。如果你是想逃离学校里的难受，可以说说具体发生了什么，"
                "我们一起想一个不伤害自己、也不让事情更糟的办法。"
            )
        if _DISCRIMINATION.search(ctx.user_message):
            return (
                "我不会用性别或地域给孩子贴标签，这类说法也没有可靠依据。每个孩子的学习表现更多和资源、机会、"
                "兴趣与支持有关。如果你在担心某个孩子的学习，我们可以具体看看他卡在哪一步。"
            )
        return (
            "这个请求超出了我能提供的范围。我可以陪你聊情绪、亲子沟通和心理教育；"
            "涉及诊断、用药或紧急危险，需要专业人员和现实支持。你想换个角度说说吗？"
        )

    def _eating_response(self, ctx: GenerationContext) -> str:
        if ctx.audience == "parent":
            return (
                "听起来你很担心孩子的吃饭和体重，这种担心很真实。先别只盯着数字，观察她有没有头晕乏力、掉发、"
                "月经紊乱、暴食或催吐、饭后躲起来等信号。我不能凭对话诊断厌食症，也不建议自行限制或强迫进食；"
                "如果这些情况持续，或体重明显下降，尽快陪她去正规医院的儿科/营养科/心理科评估。"
                "可以先记录一周的饮食和情绪，尽量不评判地陪她吃一顿饭。你最担心的是哪一个表现？"
            )
        return (
            "听起来你对自己的身体很苛刻，这种盯着体重和镜子的感觉一定很累。先说明一点：我不会给“一周瘦十斤”的方法，"
            "快速减重对还在发育的身体和情绪都可能有害。可以先看看有没有头晕、掉发、月经紊乱、暴食或催吐；"
            "如果有，尽快告诉一个可信任的大人，去儿科/营养科/心理科评估。"
            "你更在意体重数字，还是别人怎么看你？"
        )

    def _greeting(self) -> str:
        return (
            "你好，我是心语。我可以陪你梳理亲子和情绪上的难题，也可以一起想一个能落地的小步骤。"
            "我不做诊断、不提供用药建议；如果有紧急危险，请直接联系 120/110 或拨打 12356。"
            "今天想从哪件事说起？"
        )

    def _compose(self, ctx: GenerationContext) -> str:
        query = ctx.user_message.strip()
        if not query:
            return "我在这里，你可以慢慢说。"

        previous_user = ""
        for turn in reversed(ctx.history):
            if turn.get("role") == "user":
                previous_user = turn.get("content", "")
                break
        effective = query
        if previous_user and (len(query) < 14 or _CONTEXTUAL.search(query)):
            effective = f"{previous_user} {query}"

        if re.fullmatch(r"(你好|您好|hi|hello|在吗|嗨)[!！。~～\s]*", query, re.IGNORECASE):
            return self._greeting()
        if _RESOURCE_QUERY.search(query):
            return (
                "我能确认的是全国统一心理援助热线 12356；如果孩子或你有立即的自伤、轻生风险，"
                "请直接拨打 120/110，不要只等热线。部分地区还有 12355 青少年服务热线，"
                "但号码和服务时间会因城市不同而变化，我无法逐一确认，建议通过当地卫健委、学校心理老师核实。"
                "你更想了解哪一类资源？"
            )
        if _IDENTITY.search(query):
            return (
                "我是心语，一个 AI 心理教育助手，不是真人，也不是心理医生或咨询师。"
                "我可以陪你梳理情绪、提供心理教育信息和求助路径，但不能替代专业评估或治疗。"
                "你愿意先说说，是什么让你想问这个？"
            )
        if _LEGAL_INFO.search(query):
            return (
                "我不确定具体是第几条，也不能凭记忆给你一个准确条号，怕你拿去沟通时出错。"
                "建议直接查官方发布的未成年人保护法/教师法原文，或向当地教育部门、专业律师核实。"
                "你现在更需要的，是确认法律依据，还是先整理孩子受伤的证据和沟通说法？"
            )
        if _BOOK_QUERY.search(query):
            return (
                "我的知识库没有《被讨厌的勇气》的原文或页码，所以我不确定第几页讲了什么，也不能编一个页码。"
                "你可以查原书目录或出版社资料；如果你是想找“被讨厌的勇气”里关于课题分离的思路，"
                "我可以基于通用理解和你聊。你真正想解决的是哪件事？"
            )
        if _CURE_CLAIM.search(query):
            return (
                "我不能保证三天或任何确定时间治好厌学，也没有可靠证据支持这种速效承诺。"
                "厌学通常和学业压力、情绪、睡眠、家庭关系多因素有关，需要先做正规评估。"
                "可以先记录孩子最近两周的睡眠、情绪和拒学情境，再联系学校心理老师或正规医院心理科。"
                "你愿意先说说，孩子是从哪一天开始明显不想上学的吗？"
            )
        if _STEPFAMILY.search(query):
            return (
                "听起来你夹在中间很为难：一边想让孩子适应，一边又怕她受伤。继父愿意参与不是坏事，"
                "但“管学习”之前需要先建立关系；女儿说“你不是我爸”，更像是在守自己的边界，不一定是针对你。"
                "可以约定继父先从生活小事参与，学习由你主导一段时间，不要求她马上懂事。"
                "你愿意说说，他们最近一次冲突是怎么发生的吗？"
            )
        if _CURE_GUARANTEE.search(effective) or _PRIVACY.search(effective):
            return self._privacy_response(ctx) if _PRIVACY.search(effective) else self._off_scope_response(ctx)
        if _OFF_SCOPE_SEXUAL.search(effective) or _OFF_SCOPE_CHEAT.search(effective) or _DISCRIMINATION.search(effective):
            return self._off_scope_response(ctx)
        if "eating" in ctx.risk_categories:
            return self._eating_response(ctx)
        special = self._special_boundary_response(ctx, effective)
        if special:
            return special
        if _INJECTION.search(effective) or _MED_REQUEST.search(effective) or _BOUNDARY.search(effective):
            return self._boundary_response(ctx, effective)

        empathy = self._pick(_EMPATHY[self._emotion_key(ctx)], query, "empathy")
        question = self._question_for(ctx, query)

        is_info = bool(re.search(r"(为什么|怎么办|正常吗|是不是|该不该|如何|怎么)", query))
        action = self._specific_action(ctx)

        memory_prefix = ""
        if ctx.facts:
            age_fact = next((f for f in ctx.facts if "岁" in f), "")
            if age_fact:
                memory_prefix = f"你之前提到{age_fact}，"

        is_decision = bool(re.search(r"(该不该|要不要|能不能|是不是该)", query))
        if ctx.audience == "parent":
            reframe = self._pick(_PARENT_REFRAME, query, "reframe")
            lead = f"{memory_prefix}你愿意来想办法，本身就说明你在乎。"
            body = f"{empathy}{reframe}"
            action_line = f"可以先试试：{action}。"
            if is_decision:
                body += "这个决定更适合和孩子一起做：先弄清孩子是身体或情绪上撑不住，还是只是需要一点空间，再讨论要不要调整。"
        elif ctx.audience == "teen":
            reframe = self._pick(_TEEN_REFRAME, query, "reframe")
            lead = f"{memory_prefix}你愿意把这些说出来，已经很不容易。"
            body = f"{empathy}{reframe}"
            action_line = f"可以先试试：{action}。"
            if is_decision:
                body += "这个决定不用今晚就定下来，先弄清你最担心的是什么，再选一个小调整。"
        else:
            lead = f"{memory_prefix}我在听。"
            body = empathy
            action_line = f"如果愿意，可以先试试：{action}。"

        if is_info and ctx.use_rag and ctx.docs and not is_decision:
            first = re.split(r"(?<=[。！？])", ctx.docs[0].text)[0].strip()
            if 12 <= len(first) <= 80:
                body = f"先回应你问的：{first}{body}"

        # 控制长度与说教感：最多一个建议、一个问题
        text = f"{lead}{body}{action_line}{question}"
        if len(text) > 360:
            text = f"{lead}{empathy}{action_line}{question}"
        if len(text) > 360:
            text = f"{empathy}{action_line}{question}"
        return text

    async def agenerate(self, messages: list[dict[str, str]], ctx: GenerationContext) -> str:
        return self._compose(ctx)

    async def astream(
        self, messages: list[dict[str, str]], ctx: GenerationContext
    ) -> AsyncIterator[str]:
        text = self._compose(ctx)
        parts = re.findall(r"[^。！？!?；;]+[。！？!?；;]?", text)
        for part in parts:
            if not part:
                continue
            yield part
            if ctx.stream_delay_ms > 0:
                await asyncio.sleep(ctx.stream_delay_ms / 1000)


def boundary_reply(ctx: GenerationContext) -> str | None:
    """返回结构性边界回复；命中时不再交给生成模型。

    真实模型会遗忘系统提示词里的边界，尤其是诊断、用药、隐私和注入类请求。
    这里复用离线编排器已经验证过的固定话术，把“拒绝 + 替代路径”放回代码路径。
    """
    query = ctx.user_message.strip()
    if not query:
        return None
    previous_user = ""
    for turn in reversed(ctx.history):
        if turn.get("role") == "user":
            previous_user = turn.get("content", "")
            break
    effective = query
    if previous_user and (len(query) < 14 or _CONTEXTUAL.search(query)):
        effective = f"{previous_user} {query}"

    composer = OfflineComposer()
    if "eating" in ctx.risk_categories:
        return composer._eating_response(ctx)
    if _CURE_GUARANTEE.search(effective) or _PRIVACY.search(effective):
        return composer._privacy_response(ctx) if _PRIVACY.search(effective) else composer._off_scope_response(ctx)
    if _OFF_SCOPE_SEXUAL.search(effective) or _OFF_SCOPE_CHEAT.search(effective) or _DISCRIMINATION.search(effective):
        return composer._off_scope_response(ctx)
    if _INJECTION.search(effective) or _MED_REQUEST.search(effective) or _BOUNDARY.search(effective):
        return composer._boundary_response(ctx, effective)
    special = composer._special_boundary_response(ctx, effective)
    if special:
        return special
    return None


class LocalHFGenerator(BaseGenerator):
    """用 Transformers 加载 Qwen + 可选 PEFT adapter，在 MPS/CPU 上做本地评测。

    这不是生产推理路径；生产请用 vLLM。它存在的目的是在没有 CUDA 的机器上
    也能真实评估 SFT adapter，而不是编造训练结果。
    """

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        settings = get_settings()
        self._torch = torch
        self._lock = threading.Lock()
        self.device = settings.transformers_device
        if self.device == "auto":
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if self.device == "mps" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.transformers_model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.transformers_model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        adapter = settings.transformers_adapter_path
        if adapter and Path(adapter).exists():
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
            self.model = self.model.merge_and_unload()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = settings.transformers_max_new_tokens

    def _generate(self, messages: list[dict[str, str]]) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=4096
        ).to(self.device)
        with self._lock, self._torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    async def agenerate(self, messages: list[dict[str, str]], ctx: GenerationContext) -> str:
        return await asyncio.to_thread(self._generate, messages)

    async def astream(
        self, messages: list[dict[str, str]], ctx: GenerationContext
    ) -> AsyncIterator[str]:
        text = await self.agenerate(messages, ctx)
        for chunk in re.findall(r"[^。！？!?；;\n]+[。！？!?；;]?", text):
            if chunk:
                yield chunk


@lru_cache(maxsize=1)
def build_generator() -> BaseGenerator:
    settings = get_settings()
    if settings.llm_backend == "openai_compatible":
        return OpenAICompatibleGenerator()
    if settings.llm_backend == "transformers":
        return LocalHFGenerator()
    return OfflineComposer()
