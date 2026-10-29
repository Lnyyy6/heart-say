"""校验数据并生成 data card / 统计 JSON。

用法：python scripts/data_report.py [--check]
--check：发现结构错误时以非零码退出，用于 CI。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart_say.rag.knowledge import load_chunks, load_records  # noqa: E402
from heart_say.schemas import EvalCase  # noqa: E402

PHONE = re.compile(r"1[3-9]\d{9}")
ID_CARD = re.compile(r"[1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pii_hits(text: str) -> int:
    return len(PHONE.findall(text)) + len(ID_CARD.findall(text)) + len(EMAIL.findall(text))


def avg_len(values: list[int]) -> float:
    return round(statistics.mean(values), 1) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []

    source_path = ROOT / "data/knowledge/kb_v0.jsonl"
    kb = load_records(source_path if source_path.exists() else None)
    chunks = load_chunks()
    seed_sft = load_jsonl(ROOT / "data/sft/seed_sft.jsonl")
    seed_pairs = load_jsonl(ROOT / "data/preference/seed_pairs.jsonl")
    sft = load_jsonl(ROOT / "data/sft/train.jsonl")
    pairs = load_jsonl(ROOT / "data/preference/train_pairs.jsonl")
    eval_rows = load_jsonl(ROOT / "data/eval/cases.jsonl")
    redteam = load_jsonl(ROOT / "data/redteam/prompts.jsonl")

    cases: list[EvalCase] = []
    for index, row in enumerate(eval_rows, start=1):
        try:
            cases.append(EvalCase(**row))
        except Exception as exc:
            errors.append(f"eval:{index} {exc}")

    for record in sft:
        roles = [m["role"] for m in record.get("messages", [])]
        if len(roles) < 4 or roles[0] != "user" or any(roles[i] == roles[i + 1] for i in range(len(roles) - 1)):
            errors.append(f"sft:{record.get('id')} 角色/轮数异常")
    for record in pairs:
        if not all(record.get(key) for key in ("prompt", "chosen", "rejected")):
            errors.append(f"pref:{record.get('id')} 字段缺失")
    for record in redteam:
        if not record.get("prompt") or not record.get("expected_behavior"):
            errors.append(f"redteam:{record.get('id')} 字段缺失")

    pii_total = 0
    for record in sft:
        pii_total += sum(pii_hits(m.get("content", "")) for m in record.get("messages", []))
    for record in pairs:
        pii_total += pii_hits(record.get("prompt", "")) + pii_hits(record.get("chosen", ""))
    for record in eval_rows:
        pii_total += sum(pii_hits(turn) for turn in record.get("turns", []))
    for record in redteam:
        pii_total += pii_hits(record.get("prompt", ""))
    if pii_total:
        errors.append(f"检测到 {pii_total} 处疑似 PII")

    topics = Counter(record["topic"] for record in kb)
    sources = Counter(record["source_name"] for record in kb)
    sft_audience = Counter(record.get("audience", "unknown") for record in sft)
    sft_risk = Counter(record.get("risk_level", "unknown") for record in sft)
    eval_categories = Counter(case.category for case in cases)
    risk_categories = Counter(case.expected_risk for case in cases)
    redteam_categories = Counter(record.get("category", "unknown") for record in redteam)
    reject_reasons = Counter(record.get("reject_reason", "unknown") for record in pairs)
    assistant_lengths = [
        len(message["content"])
        for record in sft
        for message in record.get("messages", [])
        if message.get("role") == "assistant"
    ]
    kb_lengths = [len(record["text"]) for record in kb]

    stats = {
        "generated_on": date.today().isoformat(),
        "knowledge": {
            "documents": len(kb),
            "chunks": len(chunks),
            "avg_chars": avg_len(kb_lengths),
            "topics": dict(topics),
            "sources": dict(sources),
        },
        "sft": {
            "count": len(sft),
            "seed_count": len(seed_sft),
            "audience": dict(sft_audience),
            "risk": dict(sft_risk),
            "avg_assistant_chars": avg_len(assistant_lengths),
            "turns": dict(Counter(len(r.get("messages", [])) // 2 for r in sft)),
        },
        "preference": {
            "count": len(pairs),
            "seed_count": len(seed_pairs),
            "reject_reasons": dict(reject_reasons),
        },
        "eval": {
            "count": len(cases),
            "categories": dict(eval_categories),
            "risk": dict(risk_categories),
        },
        "redteam": {"count": len(redteam), "categories": dict(redteam_categories)},
        "pii_hits": pii_total,
        "errors": errors,
    }

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "data_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    reject_rows = "\n".join(
        f"| {name} | {count} |" for name, count in reject_reasons.most_common()
    )

    data_card = f"""# 心语数据卡（Data Card）

版本：v1 · {date.today().isoformat()}
适用范围：心理教育与亲子沟通对话助手的研究/作品集演示，不用于临床或真实患者数据。

## 1. 数据概览

| 数据 | 数量 | 说明 |
| --- | ---: | --- |
| 知识库来源卡片 | {len(kb)} | 公开心理教育材料改写，覆盖 9 个主题 |
| 检索片段 | {len(chunks)} | 来源卡 + 要点/问答/信号派生片段 |
| SFT 对话 | {len(sft)} | {len(seed_sft)} 条种子 + {len(sft) - len(seed_sft)} 条模板增强；家长 {sft_audience.get('parent', 0)} / 青少年 {sft_audience.get('teen', 0)} |
| 偏好对 | {len(pairs)} | {len(seed_pairs)} 条种子 + {len(pairs) - len(seed_pairs)} 条增强；单一缺陷标注 |
| 评测用例 | {len(cases)} | 7 类能力，含危机与记忆 |
| 红队 prompt | {len(redteam)} | 危机、用药、诊断、注入、隐私等 |
| 检测到的 PII | {pii_total} | 手机号/身份证/邮箱正则扫描 |

## 2. 知识库

- 来源卡片平均长度：{avg_len(kb_lengths)} 字；检索片段 {len(chunks)}（原始卡片保留，派生片段不新增事实）。
- 主题分布：

| 主题 | 文档数 |
| --- | ---: |
{_topic_table(topics)}

- 主要来源（公开科普材料改写，不逐字复制）：

| 来源 | 文档数 |
| --- | ---: |
{_source_table(sources)}

- 每张卡片包含 `title / topic / audience / risk_level / source / key_points / text`，检索时保留来源与要点，回答可显示引用。

## 3. SFT 对话

- 条数：{len(sft)}（{len(seed_sft)} 条人工种子 + {len(sft) - len(seed_sft)} 条本地模板增强）；平均 assistant 回复 {avg_len(assistant_lengths)} 字。
- 受众：家长 {sft_audience.get('parent', 0)}、青少年 {sft_audience.get('teen', 0)}。
- 风险分布：{json.dumps(dict(sft_risk), ensure_ascii=False)}。
- 轮数分布：{json.dumps(dict(Counter(len(r.get('messages', [])) // 2 for r in sft)), ensure_ascii=False)}。
- 覆盖场景：考试焦虑、手机冲突、校园排挤、亲子冷战、离婚家庭、二胎、睡眠、身材焦虑、厌学、自伤危机、诊断/用药越界等。
- 构造方式：种子先行，决定风格与安全边界；`scripts/augment_local_data.py` 用 21 个场景生成模板增强，assistant 回复来自离线编排器与固定安全话术，**不是强模型合成**；`scripts/synthesize_data.py` 保留强模型扩充管线。清洗脚本统一做 PII 脱敏、角色校验、去重与安全过滤。
- 使用记录：原始 60 条种子跑过 1 epoch SFT（4.33/96%）；1110 条扩充数据在 16GB M1 Pro 上跑 50 step（≈9% epoch，loss 3.08→0.64，峰值 11.56GB）后按检查点停止，100 条评测 4.60/100%，完整 1 epoch 需空闲 GPU/大内存机器。

## 4. 偏好对

- 条数：{len(pairs)}（{len(seed_pairs)} 条人工种子 + {len(pairs) - len(seed_pairs)} 条本地增强）；每条 chosen 具备共情、具体建议与边界，rejected 恰好包含一个典型缺陷。
- 拒绝原因分布：

| 原因 | 数量 |
| --- | ---: |
{reject_rows}

- 用途：DPO 对齐；危机/边界对用于降低越界承诺与不当处置。
- 使用记录：30 对种子做过 1.5B DPO（4.31/98%）；555 条增强对继续跑 60 step DPO（≈22% epoch，峰值 12.96GB），评测 4.60/100%，与 SFT 持平；rejected 不含可操作伤害方法。

## 5. 评测与红队

- 评测类别：{json.dumps(dict(eval_categories), ensure_ascii=False)}。
- 期望风险分布：{json.dumps(dict(risk_categories), ensure_ascii=False)}。
- 红队类别：{json.dumps(dict(redteam_categories), ensure_ascii=False)}。
- 评测 judge 默认为可复现规则评分，`JUDGE_BACKEND=llm` 时可切换 LLM-as-judge；上线前需人工抽检 10-20%。

## 6. 隐私与合规

- 未使用真实用户对话或真实患者数据；SFT / 评测 / 红队均为合成或公开材料改写。
- 所有数据经过手机号、身份证号、邮箱、社交账号正则脱敏；训练前 `scripts/build_training_data.py` 再清洗一次。
- 不包含可识别个人身份的信息；如后续接入真实数据，需单独做知情同意、脱敏与访问控制。
- 知识库仅用于心理教育，不构成诊断或治疗建议；危机资源为中国大陆公开热线（12356、120/110），上线前需再次核验。

## 7. 已知局限

- 知识库为 {len(kb)} 张来源卡派生 {len(chunks)} 片段，数量达标但派生片段与原文重叠，需继续去重与质量评估；真实新来源仍只有 {len(kb)} 篇。
- 1050 条扩充对话是模板 + 离线编排器生成，句式和场景重复度高于强模型合成；已人工抽看，但未做双人标注与一致性统计。扩充 SFT 的 4.60 与评测句式有重叠，可能高估泛化，不能替代独立盲评。
- 评测集 100 条，覆盖仍有限；危机场景为文本模拟，不能替代真实临床评估。
- 规则 judge 对“共情质量”的测量是代理指标，需与人工评分做相关性校准。
- 热线与转介资源可能变化，部署前需复核。

## 8. 版本与复现

```bash
python scripts/data_report.py          # 重新生成统计与数据卡
python scripts/build_training_data.py  # 清洗并生成 train.jsonl
python -m heart_say.evaluation.runner --name rag
```

数据文件：`data/knowledge/kb.jsonl`、`data/knowledge/kb_v0.jsonl`、`data/sft/train.jsonl`、
`data/preference/train_pairs.jsonl`、`data/eval/cases.jsonl`、`data/redteam/prompts.jsonl`。
"""
    (ROOT / "docs" / "DATA_CARD.md").write_text(data_card, encoding="utf-8")

    print(f"数据卡已写入 docs/DATA_CARD.md；错误 {len(errors)} 条")
    for error in errors[:20]:
        print(" -", error)
    if args.check and errors:
        raise SystemExit(1)


def _topic_table(topics: Counter) -> str:
    return "\n".join(f"| {topic} | {count} |" for topic, count in topics.most_common())


def _source_table(sources: Counter) -> str:
    return "\n".join(f"| {name} | {count} |" for name, count in sources.most_common(12))


if __name__ == "__main__":
    main()
