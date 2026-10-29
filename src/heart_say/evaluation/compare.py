"""生成 baseline vs RAG 对比报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import CATEGORY_LABELS, DIMENSION_LABELS

DIMENSIONS = ("empathy", "boundary", "safety", "hallucination", "persona", "preachiness")


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"报告不存在：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="对比两份评测报告")
    parser.add_argument("--baseline", default="reports/eval_baseline.json")
    parser.add_argument("--rag", default="reports/eval_rag.json")
    parser.add_argument("--output", default="reports/baseline_comparison.md")
    parser.add_argument("--left-label", default="")
    parser.add_argument("--right-label", default="")
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    base = _load(Path(args.baseline))
    rag = _load(Path(args.rag))
    left = args.left_label or base.get("name", "baseline")
    right = args.right_label or rag.get("name", "rag")
    title = args.title or f"{left} vs {right}"
    if base["cases"] != rag["cases"]:
        note = f"注意：两份报告用例数不同（{base['cases']} vs {rag['cases']}），仅作参考。"
    else:
        note = "两份报告使用同一用例集与同一 judge 配置。"

    lines = [
        f"# {title} 对比",
        "",
        f"- 左：`{base['name']}`，RAG {'开启' if base.get('use_rag') else '关闭'}，模型 `{base['model']}`",
        f"- 右：`{rag['name']}`，RAG {'开启' if rag.get('use_rag') else '关闭'}，模型 `{rag['model']}`",
        f"- {note}",
        f"- 用例数：{rag['cases']}",
        "",
        f"| 指标 | {left} | {right} | 变化 |",
        "| --- | ---: | ---: | ---: |",
        f"| 总均分 | {base['overall']} | {rag['overall']} | {rag['overall'] - base['overall']:+.2f} |",
        f"| 通过率 | {base['pass_rate']:.1%} | {rag['pass_rate']:.1%} | {(rag['pass_rate'] - base['pass_rate']) * 100:+.1f} pp |",
        f"| 引用覆盖率 | {base['citation_rate']:.0%} | {rag['citation_rate']:.0%} | {(rag['citation_rate'] - base['citation_rate']) * 100:+.0f} pp |",
        f"| P95 延迟 | {base['latency_ms']['p95']} ms | {rag['latency_ms']['p95']} ms | {rag['latency_ms']['p95'] - base['latency_ms']['p95']:+.1f} ms |",
        "",
        "## 维度变化",
        "",
        f"| 维度 | {left} | {right} | 变化 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in DIMENSIONS:
        b = base["dimensions"].get(name, 0)
        r = rag["dimensions"].get(name, 0)
        lines.append(f"| {DIMENSION_LABELS[name]} | {b} | {r} | {r - b:+.2f} |")

    lines.extend(
            [
                "",
                "## 分类表现",
                "",
                f"| 分类 | 用例 | {left} 均分 | {right} 均分 | 变化 |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
    rag_categories = {item["category"]: item for item in rag["categories"]}
    for item in base["categories"]:
        category = item["category"]
        other = rag_categories.get(category)
        if not other:
            continue
        lines.append(
            f"| {CATEGORY_LABELS.get(category, category)} | {other['cases']} | "
            f"{item['overall']} | {other['overall']} | {other['overall'] - item['overall']:+.2f} |"
        )

    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 从 {left} 到 {right} 的变化只说明这一组配置的差异，需要结合具体失败样本解释；",
            "- 若延迟显著上升，需要在线上用流式、缓存与更小 top_k 折中；",
            "- 自动化分数只用于回归，最终结论需结合人工抽检与安全红队。",
            "",
            "> 当前离线演示使用 hash embedding + 启发式 rerank，数字不代表 BGE-M3/Qwen 的最终效果；"
            "切换真实模型/权重后需重跑本报告。",
            "",
        ]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"对比报告已写入 {output}")


if __name__ == "__main__":
    main()
