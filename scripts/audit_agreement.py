"""计算人工抽检与 judge 的一致性（CSV 填好后运行）。

用法：python scripts/audit_agreement.py
输入：reports/human_audit_sample.csv（human_overall / human_pass / human_notes）
输出：reports/human_audit_agreement.md
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cohen_kappa(left: list[bool], right: list[bool]) -> float:
    if not left:
        return 0.0
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    p_left = sum(left) / len(left)
    p_right = sum(right) / len(right)
    expected = p_left * p_right + (1 - p_left) * (1 - p_right)
    if expected == 1:
        return 1.0
    return round((observed - expected) / (1 - expected), 3)


def main() -> None:
    path = ROOT / "reports" / "human_audit_sample.csv"
    if not path.exists():
        raise SystemExit("没有 reports/human_audit_sample.csv，请先运行 scripts/build_audit.py")
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    rated = [row for row in rows if row.get("human_overall", "").strip()]
    if not rated:
        raise SystemExit("CSV 还没有 human_overall；请人工评分后再运行。")

    diffs = []
    judge_flags, human_flags = [], []
    mismatches = []
    for row in rated:
        human = float(row["human_overall"])
        judge = float(row["judge_overall"])
        diffs.append(abs(human - judge))
        human_pass = row.get("human_pass", "").strip().lower() in ("yes", "y", "true", "1", "通过")
        judge_pass = str(row.get("judge_pass", "")).lower() in ("true", "1", "yes")
        judge_flags.append(judge_pass)
        human_flags.append(human_pass)
        if human_pass != judge_pass or abs(human - judge) >= 1:
            mismatches.append(
                {
                    "id": row["id"],
                    "judge": judge,
                    "human": human,
                    "judge_pass": judge_pass,
                    "human_pass": human_pass,
                    "notes": row.get("human_notes", "") or row.get("judge_issues", ""),
                }
            )

    agree = sum(a == b for a, b in zip(judge_flags, human_flags, strict=True))
    lines = [
        "# 人工抽检一致性",
        "",
        f"- 已评分：{len(rated)} / {len(rows)}",
        f"- 平均绝对差：{statistics.mean(diffs):.2f} 分",
        f"- 通过/不通过一致率：{agree / len(rated):.1%}",
        f"- Cohen's kappa：{cohen_kappa(judge_flags, human_flags)}",
        "",
        "## 不一致样本",
        "",
    ]
    if mismatches:
        for item in mismatches:
            lines.append(
                f"- `{item['id']}` judge {item['judge']} ({item['judge_pass']}) / "
                f"人工 {item['human']} ({item['human_pass']})：{item['notes']}"
            )
    else:
        lines.append("无。")
    lines.extend(
        [
            "",
            "> 一致性只说明 judge 与这次人工抽检的接近程度，不代表临床有效性。",
            "",
        ]
    )
    out = ROOT / "reports" / "human_audit_agreement.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
