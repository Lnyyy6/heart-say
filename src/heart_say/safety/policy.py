"""安全策略：输入风险分级、危机应答协议、输出守门与 PII 脱敏。

设计原则：
1. 输入侧规则优先保证高召回，宁可把模糊表达升级为需要确认安全；
2. 危机场景不做普通说教，先确认当下安全、连接现实支持与紧急资源；
3. 输出侧拦截诊断结论、具体用药、绝对化承诺与自伤方法细节；
4. 规则可解释、可测试，模型只作为补充而不是安全底线。
"""

from __future__ import annotations

import re
import unicodedata

from ..config import get_settings
from ..schemas import RiskAssessment, RiskLevel

DISCLAIMER = (
    "心语提供心理教育与情绪陪伴，不进行医疗诊断，也不能替代心理咨询或精神科治疗。"
    "如果你或身边的人正处于紧急危险中，请立即拨打 120/110，或联系全国心理援助热线 12356。"
)

_HOTLINE = "12356"

# (正则, 类别, 基础等级)
_RISK_RULES: list[tuple[re.Pattern[str], str, RiskLevel]] = [
    (re.compile(r"(自杀|轻生|结束自己|结束(自己的)?生命|不想活|活不下去|活着没意思|活着没意义|死了算了|想死|去死|(同学|朋友|他|她).{0,20}(以后别找我|电话不接|消息不回|最后一条))"), "suicide", "high"),
    (re.compile(r"(遗书|告别信|最后一条消息|交代后事|安排好身后|谢谢.{0,6}以后别找我)"), "suicide", "critical"),
    (re.compile(r"(跳楼|跳下去|从阳台|从楼上|从楼顶|跳桥|卧轨|上吊|割腕|烧炭|吞药|吞了.{0,4}药|喝农药|吃了.{0,4}(很多|大量).{0,2}药|多吃.{0,4}片.{0,8}(睡过去|不醒)|站在.{0,4}(阳台|天台|楼顶)|(攒|存|囤).{0,4}药|药.{0,4}(一起|都).{0,2}(吃|吞))"), "suicide", "critical"),
    (re.compile(r"(自残|自伤|伤害自己|(割|划)(伤|破|了)?(胳膊|手腕|手臂)|拿刀划|烫自己|掐自己|怎么划|不留疤|划痕|用烫的|烟头烫|控制.{0,4}力度.{0,6}(烫|划|伤)|血.{0,4}止不住|大量出血)"), "self_harm", "high"),
    (re.compile(r"(不想吃饭|吃不下|暴食|催吐|厌食|催泻|体重掉得很快|怎么.{0,4}(瘦|减肥)|一周瘦|快速减重|怕胖|觉得自己胖|90斤)"), "eating", "medium"),
    (re.compile(r"(霸凌|欺凌|校园暴力|收保护费|勒索|(堵|围住|拦).{0,12}(要钱|威胁)|威胁.{0,6}(不许|不能).{0,4}(说|告诉)|网暴|人肉|扒.{0,4}(学校|住址)|评论区.{0,4}骂|起外号|被孤立)"), "bullying", "high"),
    (re.compile(r"(家暴|打我|被打|殴打|虐待|性侵|猥亵|摸我|被骚扰|性骚扰|被.{0,8}(摸|碰).{0,4}(身体|下面|隐私部位)|表哥.{0,6}摸|强迫我|(继父|继母).{0,10}(摸|打|侵犯|猥亵)|骂我.{0,6}(废物|去死)|语言暴力|长期羞辱|(老师|教练|家教)(?!(不能|不应|禁止|不得)).{0,8}(打|体罚|扇|尺子)|被(老师|教练|家教).{0,6}(打|体罚|扇))"), "abuse", "high"),
    (re.compile(r"(想杀|弄死|报复|带刀|捅|下药|同归于尽|找.{0,6}算账|网暴|社死|曝光.{0,4}隐私|打回去|教我怎么下手)"), "violence", "high"),
    (re.compile(r"(吃几片|吃多少片|一天吃多少|什么药|吃什么药|过量|安眠药|褪黑素|舍曲林|氟西汀|帕罗西汀|阿普唑仑|劳拉西泮|奥氮平|利培酮|锂盐|药量|开点药|停药|自己停|减药|加药|减半|加量|两种药.{0,6}(一起|同吃))"), "medication", "medium"),
    (re.compile(r"(我是不是|是不是得了|算不算|确诊|诊断|确认一下|给个结论|判断一下)(抑郁|焦虑|多动|双相|自闭|强迫|ADHD|精神病)"), "diagnosis", "medium"),
    (re.compile(r"(孩子|他|她).{0,6}(抑郁|焦虑症|多动症|双相|自闭症|精神病)"), "diagnosis", "medium"),
    (re.compile(r"(量表|测评|自测|测试结果).{0,12}(抑郁|焦虑|双相|多动|ADHD)"), "diagnosis", "medium"),
]

_IMMINENT = re.compile(
    r"(现在|今晚|今天|马上|立刻|已经|正在|准备好了|刀|楼顶|天台|绳子|药在手里|吞了|割了|"
    r"不想等到|最后一次|撑不过)"
)
_NEGATION = re.compile(r"(没有|没|不会|从没|从来没有|并不|不是).{0,5}(自杀|自伤|轻生|想死)")
_THIRD_PERSON = re.compile(r"(孩子|女儿|儿子|娃|学生|同学|朋友|他|她)")
_THIRD_SUBJECT = re.compile(r"(我(的)?(朋友|同学|孩子|女儿|儿子|学生|同桌)|朋友|同学|孩子|女儿|儿子)")
_SELF_SUBJECT = re.compile(
    r"(我(想|要|已经|正在|刚|刚刚)?(自杀|自伤|自残|割腕|割了|吞药|跳楼|轻生|吃了.{0,4}药|站在.{0,4}(阳台|天台|楼顶))|"
    r"我不想活|我想死|我难受的时候会划|我控制不住|我想(找人|打回去|带刀|报复|杀|弄死))"
)
_SELF_REFERENCE = re.compile(r"(我|自己)")
_PARENT_HINT = re.compile(r"(我(的)?(孩子|女儿|儿子|娃)|当妈|当爸|家长|我家孩子|孩子他|作为妈妈|作为爸爸)")
_TEEN_HINT = re.compile(r"(我(妈|爸|父母)|我妈|我爸|上学|同学|老师|我(今年)?\s*1[0-9]\s*岁|我(今年)?\s*[1-9]\s*岁)")

PARENT = "parent"
TEEN = "teen"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", "", text)


_TEEN_EXPLICIT = re.compile(r"(我(妈|爸|父母)|我妈|我爸|我(今年)?\s*\d{1,2}\s*岁)")
_CHILD_SUBJECT = re.compile(r"(孩子|女儿|儿子|娃)")
_TEEN_EXAM = re.compile(r"我.{0,6}(考试|考砸|月考|期末|中考|高考|成绩)")


def detect_audience(text: str, fallback: str = "unknown") -> str:
    text = normalize(text)
    # 明确线索优先：家长称呼 > 青少年自述 > 只提“孩子” > 上学/考试等弱线索
    if _PARENT_HINT.search(text):
        return PARENT
    if _TEEN_EXPLICIT.search(text):
        return TEEN
    if _CHILD_SUBJECT.search(text):
        return PARENT
    if _TEEN_EXAM.search(text):
        return TEEN
    if _TEEN_HINT.search(text):
        return TEEN
    return fallback  # type: ignore[return-value]


def assess_risk(text: str, audience: str = "unknown") -> RiskAssessment:
    """对单条用户输入做可解释的风险评估。"""
    text = normalize(text)
    if not text:
        return RiskAssessment(level="none", action="none")

    # 询问“怎么/方法/能不能”时，否定词不等于否认风险，例如“有没有不留疤的自伤方法”
    asking = bool(re.search(r"(方法|怎么|如何|教|能不能|可不可以|吗|多久|哪里)", text))
    if _NEGATION.search(text) and not asking:
        # 明确否认时降级，但保留后续对话观察
        return RiskAssessment(
            level="low",
            categories=[],
            action="support",
            reason="存在风险词但被明确否认，保持关注",
        )

    hits: list[tuple[str, RiskLevel]] = []
    for pattern, category, level in _RISK_RULES:
        if pattern.search(text):
            hits.append((category, level))

    if not hits:
        return RiskAssessment(level="none", action="none", reason="未命中风险规则")

    order = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    level: RiskLevel = max((h[1] for h in hits), key=lambda x: order[x])
    categories = sorted({h[0] for h in hits})

    # 情境升级：明确时间/手段/已发生行为
    if level in ("high", "critical") and _IMMINENT.search(text):
        level = "critical"

    # 第三方主语（家长描述孩子）不直接视为用户自伤，但仍需协议式回应
    is_third_person = bool(_THIRD_PERSON.search(text)) and not _SELF_REFERENCE.search(text)
    action = "protocol" if level in ("high", "critical") else "support"
    reason = f"命中 {','.join(categories)}，等级 {level}"
    if is_third_person:
        reason += "；主语为第三方，按帮助者场景回应"

    return RiskAssessment(level=level, categories=categories, action=action, reason=reason)


def resources_for(level: RiskLevel) -> list[dict[str, str]]:
    if level not in ("high", "critical"):
        return []
    return [
        {
            "name": "全国心理援助热线",
            "contact": _HOTLINE,
            "description": "24 小时心理支持（中国大陆）",
        },
        {"name": "紧急救援", "contact": "120 / 110", "description": "有立即生命危险时拨打"},
        {
            "name": "身边的支持者",
            "contact": "家人、老师、学校心理老师",
            "description": "尽量不独处，让可信任的人陪着你",
        },
    ]


def _is_third_person(text: str) -> bool:
    if _SELF_SUBJECT.search(text):
        return False
    if _THIRD_SUBJECT.search(text):
        return True
    return bool(_THIRD_PERSON.search(text)) and not _SELF_REFERENCE.search(text)


def crisis_response(text: str, risk: RiskAssessment, audience: str = "unknown") -> str:
    """生成危机/高风险的首轮安全应答。不调用大模型，避免生成不稳定。"""
    categories = set(risk.categories)
    normalized = normalize(text)
    third = _is_third_person(normalized)
    helper = (
        third
        and audience != PARENT
        and not _PARENT_HINT.search(normalized)
        and not re.search(r"(孩子|女儿|儿子|娃)", normalized)
    )
    level = risk.level

    if helper:
        return (
            "听起来你很担心朋友的安全，这份在意很重要。你不是他的治疗者，也不该独自扛这件事。"
            "先确认他此刻是否安全、身边有没有大人；如果他已经在做危险的事，请立即拨打 120/110。"
            "不要答应替他保守秘密：可以告诉他“我不会到处说，但你需要有人陪着”，并一起联系他信任的大人、"
            "老师或学校心理老师。你也可以拨打 12356，先问问专业人员怎么处理。"
            "你现在能联系到他吗？他身边有人吗？"
        )

    if "bullying" in categories:
        if third or audience == PARENT:
            return (
                "孩子被堵、被要钱，这已经是需要大人介入的欺凌，不是孩子之间的小摩擦。"
                "先确认他有没有受伤、此刻是否安全，保留聊天记录、伤情照片等证据，今天尽快联系班主任和学校负责人，"
                "要求书面处理；如果涉及勒索、威胁或人身伤害，可以拨打 110，受伤请拨打 120。"
                "你气得想冲过去很正常，但直接对峙可能让孩子更害怕，也容易让你陷入麻烦。"
                "回家先告诉他“这不是你的错，我会保护你”。你也可以拨打 12356，获得家长和孩子的专业支持。"
                "你现在能确认孩子是安全的吗？"
            )
        return (
            "被堵、被要钱不是你的错，也不是你不够强。先确保自己安全：尽量不单独去厕所或偏僻的地方，"
            "把时间和经过记下来，告诉一个可信任的大人（家长、老师或学校心理老师）。"
            "如果对方威胁、动手或继续勒索，可以拨打 110；如果身体受伤，拨打 120。"
            "你也可以拨打 12356 获得心理支持。这不是让你一个人扛的事。你现在安全吗？"
        )

    if "violence" in categories and (third or audience == PARENT):
        return (
            "我听见你现在气到想直接去找对方，那种想保护孩子的冲动很真实。先别在气头上行动，"
            "因为对峙可能让孩子更危险，也可能让你自己受伤或惹上麻烦。"
            "先确认孩子此刻是否安全、有没有受伤，把证据留好，联系学校负责人和班主任处理；"
            "涉及威胁、勒索或人身伤害，请拨打 110，受伤请拨打 120。你也可以拨打 12356 获得支持。"
            "你现在最担心孩子哪一点？"
        )

    if "violence" in categories:
        return (
            "我听见你真的很愤怒，甚至想过用伤害对方的方式解决。先离开让你冲动的现场，别让这一下的决定改变你的人生。"
            "如果你已经准备动手，请立刻停下并联系一个可信任的大人；有人身危险请拨打 120/110。"
            "你也可以拨打 12356，把这份愤怒先说出来。你愿意告诉我，刚才发生了什么吗？"
        )

    if "abuse" in categories:
        if third or audience == PARENT:
            return (
                "你提到的情况让我很担心孩子的安全。请先把孩子的安全放在第一位：如果孩子正处于危险中，"
                "立即拨打 110/120，或尽快去急诊，并让孩子远离施害者、待在有人陪伴的地方。"
                "这不是孩子的错，也不需要替对方保密。你可以联系学校心理老师、未成年人保护相关机构，"
                "或拨打全国心理援助热线 12356 获得专业支持。"
                "关于能不能告、责任怎么认定，我不能替你下法律结论，建议咨询专业律师或当地教育部门；"
                "先保留伤情照片、就医记录和沟通证据。你现在能确保孩子是安全的吗？"
            )
        return (
            "谢谢你愿意说出来，这不是你的错。现在最重要的是你的安全：先去一个施害者找不到、"
            "有可信任的人在的地方，联系你信任的大人、老师或直接拨打 110。"
            "如果身体受伤或处于危险中，请拨打 120。你也可以拨打全国心理援助热线 12356。"
            "我不会要求你独自承担，也不会让你承诺保密。你现在安全吗？身边有没有可以马上联系的人？"
        )

    if third or audience == PARENT:
        return (
            "我听见你现在非常担心孩子的安全，这种害怕是很真实的。先做三件事：确认孩子此刻在哪里、"
            "是否安全；尽量陪在孩子身边，温和而直接地问一句“你最近有没有想过伤害自己”；"
            "把家里的药物、利器先收好。若孩子已经有明确计划或危险行为，请立即拨打 120/110，或尽快去急诊，"
            "并联系学校心理老师或精神科/心理科做专业评估。你也可以拨打全国心理援助热线 12356。"
            "你现在能联系到孩子吗？"
        )

    if level == "critical":
        return (
            "我很在意你刚才说的，也听见你现在可能已经很难撑下去了。此刻先不一个人扛："
            "请马上离开可能伤害自己的地方或物品，去有人的地方，联系一个你信任的大人陪着你。"
            "如果你已经有具体计划或正在受伤，请立即拨打 120 或 110，或尽快去急诊。"
            "你也可以拨打全国心理援助热线 12356，24 小时都有人接听。"
            "我不能替代紧急救援，也不承诺保密；如果你已经吃了药，请立刻拨打 120，"
            "告诉急救人员吃了什么、多少、什么时候。你需要现实中的专业支持，不必一个人扛。"
            "你现在安全吗？身边有没有可以马上联系的人？"
        )

    return (
        "谢谢你把这些告诉我，能说出来需要很大的力气。听起来你现在被压得很难受，"
        "而且可能已经想过伤害自己。我想先确认你的安全：你现在有具体的打算吗？身边有没有药、刀或其他危险物品？"
        "如果答案是“有”，请先离开它们，去一个有大人在的地方，并马上联系你信任的人。"
        "有生命危险请拨打 120/110，或尽快去急诊；也可以拨打全国心理援助热线 12356 获得专业支持。"
        "我不会用一句“会好起来的”敷衍你，我们可以先把今晚怎么安全度过说清楚。"
    )


# ---------------- 输出守门 ----------------

_DRUGS = re.compile(
    r"(舍曲林|氟西汀|帕罗西汀|氟伏沙明|西酞普兰|艾司西酞普兰|文拉法辛|度洛西汀|米氮平|"
    r"阿普唑仑|劳拉西泮|氯硝西泮|地西泮|唑吡坦|奥氮平|利培酮|喹硫平|阿立哌唑|"
    r"丙戊酸|拉莫三嗪|碳酸锂|锂盐|哌甲酯|托莫西汀)"
)
_DOSAGE = re.compile(r"\d+(\.\d+)?\s*(mg|毫克|片|粒|ml|毫升)")
_DIAGNOSIS = re.compile(
    r"((你|孩子|他|她|您).{0,4}(就是|得了|患有|属于|一定是|确诊为|是典型的?)|诊断(为|是)).{0,6}"
    r"(抑郁症|焦虑症|双相情感障碍|双相|多动症|ADHD|自闭症|孤独症|精神分裂|强迫症|"
    r"进食障碍|创伤后应激障碍|PTSD|人格障碍)"
)
_METHOD = re.compile(
    r"(割(多深|哪里|几刀)|吃(多少|几)片|剂量(是|为)|上吊(方法|怎么)|"
    r"烧炭(步骤|怎么)|跳(哪|怎么跳)|哪种药(能|可以)(死|结束))"
)
_GUARANTEE = re.compile(
    r"(我保证|一定能|肯定会|包治|根治|(?<!不)(?<!不能)(?<!无法)(?<!没法)"
    r"保证(你|孩子)?(会|能)?(好|治好|没事|不会))"
)
_SECRECY = re.compile(
    r"((?<!能保证)绝对保密|(?<!能)保证保密|不会告诉任何人|只属于我们之间)"
)
_PREACHY = re.compile(r"(你应该|你必须|你一定要|你不能|你要记住|作为(一个)?(家长|父母)|不许|不准)")


def guard_output(text: str, risk_level: RiskLevel = "none") -> tuple[str, list[str]]:
    """拦截越界内容。返回 (处理后文本, 命中标记)。"""
    if not get_settings().enable_output_guard:
        return text, []

    flags: list[str] = []
    sentences = re.split(r"(?<=[。！？!?\n])", text)
    kept: list[str] = []
    for sentence in sentences:
        stripped = sentence.strip()
        if not stripped:
            continue
        if _DRUGS.search(stripped) or _DOSAGE.search(stripped):
            flags.append("medication_advice")
            if "以下是建议" not in text:
                kept.append("具体药物和剂量需要由精神科医生面诊评估，我不能提供用药建议。")
            continue
        if _DIAGNOSIS.search(stripped):
            flags.append("diagnosis_claim")
            kept.append("仅凭对话无法做诊断，具体情况需要由有资质的专业人员评估。")
            continue
        if _METHOD.search(stripped):
            flags.append("harm_method")
            kept.append("关于伤害方式的具体细节我不展开，我们更关心此刻怎么确保安全。")
            continue
        if _GUARANTEE.search(stripped):
            flags.append("false_guarantee")
            kept.append("我没办法保证结果，但可以陪你一起把下一步变小、变具体。")
            continue
        if _SECRECY.search(stripped):
            flags.append("secrecy_promise")
            kept.append("我会尽量保护你的隐私，但涉及人身安全时，需要联系现实中可信任的大人。")
            continue
        kept.append(sentence)

    result = "".join(kept).strip()
    if _PREACHY.search(result):
        flags.append("preachy_tone")

    if risk_level in ("high", "critical") and _HOTLINE not in result:
        result += f"\n\n如果此刻有危险，请拨打 120/110；全国心理援助热线 {_HOTLINE} 也 24 小时有人接听。"

    return result, sorted(set(flags))


# ---------------- PII ----------------

_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"1[3-9]\d{9}"), "[手机号]"),
    (re.compile(r"[1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]"), "[身份证号]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[邮箱]"),
    (re.compile(r"(QQ|微信|WeChat|VX|vx)[:：]?\s*[A-Za-z0-9_-]{5,}"), "[社交账号]"),
    (re.compile(r"(\d{1,3}\.){3}\d{1,3}"), "[IP地址]"),
]


def redact_pii(text: str) -> str:
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text
