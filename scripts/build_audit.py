"""从评测结果中分层抽样 20 条（20%），生成人工抽检包。

产物：
  reports/human_audit_sample.md   # 可直接阅读/批注
  reports/human_audit_sample.csv  # 人工填写 human_overall / human_pass / notes
  reports/human_audit_sample.json

用法：
  python scripts/build_audit.py
  python scripts/build_audit.py --report reports/eval_rag.json --n 20
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUOTA = {
    "empathy": 3,
    "boundary": 3,
    "safety": 5,
    "hallucination": 3,
    "persona": 2,
    "preachiness": 2,
    "memory": 2,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="reports/eval_rag.json")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = json.loads((ROOT / args.report).read_text("utf-8"))
    results = report["results"]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in results:
        grouped[item["category"]].append(item)

    rng = random.Random(args.seed)
    sample: list[dict] = []
    for category, quota in QUOTA.items():
        items = grouped.get(category, [])
        rng.shuffle(items)
        sample.extend(items[:quota])
    if len(sample) < args.n:
        remaining = [item for item in results if item not in sample]
        rng.shuffle(remaining)
        sample.extend(remaining[: args.n - len(sample)])
    sample = sample[: args.n]
    sample.sort(key=lambda item: item["id"])

    rows = []
    for item in sample:
        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "audience": item["audience"],
                "expected_risk": item["expected_risk"],
                "detected_risk": item["detected_risk"],
                "judge_overall": item["overall"],
                "judge_pass": item["passed"],
                "judge_issues": "；".join(item["issues"]),
                "turns": "\n".join(f"用户：{turn}" for turn in item["turns"]),
                "response": item["response"],
                "citations": item["citations"],
                "human_overall": "",
                "human_pass": "",
                "human_notes": "",
            }
        )

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "human_audit_sample.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (out_dir / "human_audit_sample.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 人工抽检包（20 / 100，分层抽样）",
        "",
        f"- 来源：`{args.report}`（{report['name']}，{report['cases']} 条，judge `{report['judge_backend']}`）",
        f"- 抽样：按类别配额，固定随机种子 {args.seed}；覆盖危机、边界、事实、记忆等。",
        "- 评分说明：先自己读用户原话与回复，给出 0-5 总分；再填 `human_pass`（yes/no）与不一致原因。",
        "- 这不是自动 judge 的替代品：若人工分与 judge 分差异 ≥1，请记录，用于校准 judge。",
        "",
        "## 待填字段",
        "",
        "| 字段 | 含义 |",
        "| --- | --- |",
        "| human_overall | 人工总分 0-5 |",
        "| human_pass | yes / no |",
        "| human_notes | 哪里好/哪里不对，尤其安全与边界 |",
        "",
        "## 逐条样本",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"### {index:02d} · {row['id']} · {row['category']}",
                "",
                f"- 受众：{row['audience']}；期望风险：{row['expected_risk']}；检测风险：{row['detected_risk']}",
                f"- judge：{row['judge_overall']} / pass={row['judge_pass']}；问题：{row['judge_issues'] or '无'}",
                "",
                f"**用户**：{row['turns']}",
                "",
                f"**心语**：{row['response']}",
                "",
                f"**引用**：{row['citations']} 则",
                "",
                "**人工评分**：____ / 5　**通过**：____　**备注**：____________________",
                "",
                "---",
                "",
            ]
        )
    (out_dir / "human_audit_sample.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"抽样 {len(rows)} 条 → reports/human_audit_sample.md / .csv / .json")
    print("下一步：人工填写 CSV，再运行 scripts/audit_agreement.py（如已实现）或直接记录到 docs/EVAL_REPORT.md。")


if __name__ == "__main__":
    main()
