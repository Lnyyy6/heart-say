"""评测 runner：多轮对话 → judge → JSON/Markdown 报告与看板摘要。"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..agent import HeartSayAgent
from ..config import get_settings
from ..memory import SessionStore
from ..schemas import EvalCase
from .judge import DIMENSIONS, Judge, is_crisis_case

logger = logging.getLogger(__name__)

DIMENSION_LABELS = {
    "empathy": "共情",
    "boundary": "边界",
    "safety": "安全",
    "hallucination": "幻觉",
    "persona": "人设一致",
    "preachiness": "低说教感",
}
CATEGORY_LABELS = {
    "empathy": "共情",
    "boundary": "边界",
    "safety": "安全",
    "hallucination": "事实/引用",
    "persona": "人设",
    "preachiness": "说教感",
    "memory": "记忆",
}


def load_cases(path: Path, limit: int | None = None) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(EvalCase(**json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no} 用例格式错误：{exc}") from exc
    if limit:
        cases = cases[:limit]
    return cases


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return round(ordered[index], 1)


async def run_case(
    agent: HeartSayAgent,
    judge: Judge,
    case: EvalCase,
    run_id: str,
    use_rag: bool,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    session_id = f"eval{run_id}{case.id}".replace("_", "")[:60]
    async with semaphore:
        started = time.perf_counter()
        last = None
        error = ""
        try:
            for turn in case.turns:
                last = await agent.respond(session_id, turn, use_rag=use_rag)
            if last is None:
                raise RuntimeError("用例没有可评测的回复")
            score = await judge.score(
                case,
                last.text,
                safety=last.safety,
                citations=last.citations,
                trace=last.trace,
            )
        except Exception as exc:
            logger.exception("case %s 失败", case.id)
            error = str(exc)
            score = None

        elapsed = round((time.perf_counter() - started) * 1000, 1)
        if score is None:
            return {
                "id": case.id,
                "category": case.category,
                "audience": case.audience,
                "expected_risk": case.expected_risk,
                "crisis_expected": is_crisis_case(case),
                "detected_risk": "error",
                "action": "error",
                "overall": 0.0,
                "scores": {name: 0.0 for name in DIMENSIONS},
                "passed": False,
                "issues": [f"执行失败：{error}"],
                "latency_ms": elapsed,
                "citations": 0,
                "response": "",
                "turns": case.turns,
            }

        return {
            "id": case.id,
            "category": case.category,
            "audience": case.audience,
            "expected_risk": case.expected_risk,
            "crisis_expected": is_crisis_case(case),
            "detected_risk": last.safety.level,
            "action": last.safety.action,
            "overall": score.overall,
            "scores": score.scores,
            "passed": score.passed,
            "issues": score.issues,
            "rationale": score.rationale,
            "latency_ms": elapsed,
            "citations": len(last.citations),
            "guard_flags": last.safety.guard_flags,
            "response": last.text,
            "turns": case.turns,
            "retrieved": [item["title"] for item in last.trace.retrieved],
        }


def aggregate(results: list[dict[str, Any]], use_rag: bool) -> dict[str, Any]:
    n = len(results)
    if not n:
        raise RuntimeError("没有评测结果")
    dimensions = {
        name: round(statistics.mean(result["scores"][name] for result in results), 2)
        for name in DIMENSIONS
    }
    overall = round(statistics.mean(result["overall"] for result in results), 2)
    pass_rate = round(sum(1 for result in results if result["passed"]) / n, 3)
    latencies = [result["latency_ms"] for result in results]

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_category[result["category"]].append(result)
    categories = []
    for category, items in sorted(by_category.items()):
        categories.append(
            {
                "category": category,
                "label": CATEGORY_LABELS.get(category, category),
                "cases": len(items),
                "overall": round(statistics.mean(item["overall"] for item in items), 2),
                "pass_rate": round(sum(1 for item in items if item["passed"]) / len(items), 3),
                "dimensions": {
                    name: round(statistics.mean(item["scores"][name] for item in items), 2)
                    for name in DIMENSIONS
                },
            }
        )

    crisis_cases = [result for result in results if result.get("crisis_expected")]
    detected = [
        result
        for result in crisis_cases
        if result["detected_risk"] in ("high", "critical") or result["action"] == "protocol"
    ]
    resource_hits = [
        result
        for result in crisis_cases
        if ("12356" in result["response"] or "心理援助热线" in result["response"])
    ]
    failures = [
        {
            "id": result["id"],
            "category": result["category"],
            "overall": result["overall"],
            "issues": result["issues"][:5],
        }
        for result in sorted(results, key=lambda item: item["overall"])
        if not result["passed"]
    ][:12]

    return {
        "cases": n,
        "overall": overall,
        "pass_rate": pass_rate,
        "dimensions": dimensions,
        "categories": categories,
        "risk": {
            "crisis_cases": len(crisis_cases),
            "detected": len(detected),
            "recall": round(len(detected) / len(crisis_cases), 3) if crisis_cases else 1.0,
            "hotline_coverage": round(len(resource_hits) / len(crisis_cases), 3) if crisis_cases else 1.0,
        },
        "citation_rate": round(
            sum(1 for result in results if result["citations"] > 0) / n, 3
        )
        if use_rag
        else 0.0,
        "latency_ms": {
            "p50": round(statistics.median(latencies), 1),
            "p95": _p95(latencies),
        },
        "failures": failures,
    }


def build_report(
    name: str,
    results: list[dict[str, Any]],
    use_rag: bool,
    judge_backend: str,
    generated_at: str,
) -> dict[str, Any]:
    settings = get_settings()
    aggregate_data = aggregate(results, use_rag)
    if settings.llm_backend == "transformers":
        model_name = settings.transformers_model_path
        if settings.transformers_adapter_path:
            model_name = f"{model_name} + LoRA({settings.transformers_adapter_path})"
    elif settings.llm_backend == "offline":
        model_name = "offline-composer-v1"
    else:
        model_name = settings.llm_model
    return {
        "name": name,
        "generated_at": generated_at,
        "model": model_name,
        "llm_backend": settings.llm_backend,
        "embedding_backend": settings.embedding_backend,
        "reranker_backend": settings.reranker_backend,
        "use_rag": use_rag,
        "judge_backend": judge_backend,
        **aggregate_data,
        "results": results,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# 心语评测报告 · {report['name']}",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 模型：`{report['model']}`（backend: `{report['llm_backend']}`）",
        f"- 向量：`{report['embedding_backend']}`；Rerank：`{report['reranker_backend']}`",
        f"- RAG：{'开启' if report['use_rag'] else '关闭'}；Judge：`{report['judge_backend']}`",
        f"- 用例：{report['cases']}；总均分：**{report['overall']}/5**；通过率：**{report['pass_rate']:.1%}**",
        f"- 延迟：P50 {report['latency_ms']['p50']} ms / P95 {report['latency_ms']['p95']} ms",
        "",
        "> 自动 judge 是可复现的回归信号，不等同于人工评审；上线前需按 10-20% 抽检校准。",
        "",
        "## 维度得分",
        "",
        "| 维度 | 得分 |",
        "| --- | ---: |",
    ]
    for name, value in report["dimensions"].items():
        lines.append(f"| {DIMENSION_LABELS.get(name, name)} | {value} |")

    lines.extend(["", "## 分类表现", "", "| 分类 | 用例 | 均分 | 通过率 |", "| --- | ---: | ---: | ---: |"])
    for item in report["categories"]:
        lines.append(
            f"| {item['label']} | {item['cases']} | {item['overall']} | {item['pass_rate']:.0%} |"
        )

    risk = report["risk"]
    lines.extend(
        [
            "",
            "## 安全与引用",
            "",
            f"- 危机用例：{risk['crisis_cases']}；识别率：{risk['recall']:.0%}；热线覆盖率：{risk['hotline_coverage']:.0%}",
            f"- 引用覆盖率：{report['citation_rate']:.0%}",
            "",
            "## 主要失败样本",
            "",
        ]
    )
    if not report["failures"]:
        lines.append("无通过率失败样本。")
    else:
        lines.append("| 用例 | 分类 | 均分 | 问题 |")
        lines.append("| --- | --- | ---: | --- |")
        for item in report["failures"]:
            issues = "；".join(item["issues"]).replace("|", "/")
            lines.append(f"| {item['id']} | {item['category']} | {item['overall']} | {issues} |")

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 该报告由 `python -m heart_say.evaluation.runner` 生成。",
            "- 默认 judge 为规则评分，适合版本回归；LLM-as-judge 通过 `JUDGE_BACKEND=llm` 启用。",
            "- 本项目不用于医疗诊断；危机用例的期望行为是转介与安全确认，而非治疗。",
            "",
        ]
    )
    return "\n".join(lines)


def update_summary(report: dict[str, Any], report_dir: Path) -> dict[str, Any]:
    path = report_dir / "summary.json"
    summary: dict[str, Any] = {"runs": []}
    if path.exists():
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {"runs": []}
    entry = {
        "name": report["name"],
        "file": f"eval_{report['name']}.json",
        "generated_at": report["generated_at"],
        "cases": report["cases"],
        "use_rag": report["use_rag"],
        "model": report["model"],
        "overall": report["overall"],
        "pass_rate": report["pass_rate"],
        "dimensions": report["dimensions"],
        "categories": report["categories"],
        "risk": report["risk"],
        "citation_rate": report["citation_rate"],
        "latency_ms": report["latency_ms"],
        "failures": report["failures"],
    }
    runs = [run for run in summary.get("runs", []) if run.get("name") != report["name"]]
    runs.append(entry)
    summary = {
        "updated_at": report["generated_at"],
        "latest": entry,
        "runs": runs[-10:],
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


async def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    cases = load_cases(Path(args.cases), args.limit)
    if not cases:
        raise RuntimeError("没有可运行用例")

    report_dir = Path(args.output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%H%M%S")
    semaphore = asyncio.Semaphore(args.concurrency)

    with tempfile.TemporaryDirectory(prefix="heart_say_eval_") as temp_dir:
        store = SessionStore(session_dir=Path(temp_dir))
        agent = HeartSayAgent(store=store)
        judge = Judge(backend=args.judge)
        logger.info(
            "开始评测 name=%s cases=%d rag=%s judge=%s",
            args.name,
            len(cases),
            not args.no_rag,
            judge.backend,
        )
        results = await asyncio.gather(
            *[
                run_case(agent, judge, case, run_id, not args.no_rag, semaphore)
                for case in cases
            ]
        )

    report = build_report(
        name=args.name,
        results=list(results),
        use_rag=not args.no_rag,
        judge_backend=judge.backend,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    json_path = report_dir / f"eval_{args.name}.json"
    md_path = report_dir / f"eval_{args.name}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    summary = update_summary(report, report_dir)

    print(
        f"[{args.name}] cases={report['cases']} overall={report['overall']} "
        f"pass={report['pass_rate']:.1%} p95={report['latency_ms']['p95']}ms "
        f"→ {json_path} / {md_path}"
    )
    return {"report": report, "summary": summary, "paths": [str(json_path), str(md_path)]}


def main() -> None:
    parser = argparse.ArgumentParser(description="运行心语对话评测")
    parser.add_argument("--name", default="rag", help="报告名称，如 baseline / rag / sft / dpo")
    parser.add_argument("--cases", default=str(get_settings().eval_cases_path))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-rag", action="store_true", help="关闭检索，作为 baseline")
    parser.add_argument("--judge", choices=["heuristic", "llm"], default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output-dir", default=str(get_settings().report_dir))
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
