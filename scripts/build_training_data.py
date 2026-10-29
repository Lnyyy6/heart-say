"""合并/清洗种子数据与合成数据，产出可直接训练的 JSONL。

- 去除 PII；
- 过滤含具体药物、剂量、自伤方法的 SFT 样本；
- 去重、校验角色交替；
- 输出 data/sft/train.jsonl 与 data/preference/train_pairs.jsonl。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart_say.safety import redact_pii  # noqa: E402

DRUG = re.compile(
    r"(舍曲林|氟西汀|帕罗西汀|氟伏沙明|西酞普兰|文拉法辛|度洛西汀|米氮平|"
    r"阿普唑仑|劳拉西泮|氯硝西泮|地西泮|唑吡坦|奥氮平|利培酮|喹硫平|阿立哌唑|"
    r"丙戊酸|拉莫三嗪|碳酸锂|哌甲酯|托莫西汀)"
)
DOSAGE = re.compile(r"\d+(\.\d+)?\s*(mg|毫克|片|粒)")
METHOD = re.compile(r"(割(腕|多深|哪里)|吃(多少|几)片|上吊(方法|怎么)|烧炭(步骤|怎么)|跳楼(方法|怎么))")
DIAGNOSIS = re.compile(r"((你|孩子|他|她)就是|确诊为|得了)(抑郁症|焦虑症|双相|多动症|自闭症|精神分裂)")


def redact_messages(messages: list[dict]) -> list[dict]:
    return [
        {"role": item["role"], "content": redact_pii(item["content"]).strip()}
        for item in messages
    ]


def unsafe_text(text: str) -> bool:
    return bool(DRUG.search(text) or DOSAGE.search(text) or METHOD.search(text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", nargs="+", default=["data/sft/seed_sft.jsonl", "data/sft/synth_sft.jsonl"])
    parser.add_argument("--pairs", nargs="+", default=["data/preference/seed_pairs.jsonl", "data/preference/synth_pairs.jsonl"])
    parser.add_argument("--sft-out", default="data/sft/train.jsonl")
    parser.add_argument("--pairs-out", default="data/preference/train_pairs.jsonl")
    args = parser.parse_args()

    sft_records, seen, dropped = [], set(), Counter()
    for raw in args.sft:
        path = Path(raw)
        if not path.exists():
            continue
        for line in path.open("r", encoding="utf-8"):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = redact_messages(record.get("messages", []))
            if len(messages) < 4:
                dropped["too_short"] += 1
                continue
            if [m["role"] for m in messages][0] != "user":
                dropped["bad_roles"] += 1
                continue
            if any(unsafe_text(m["content"]) for m in messages if m["role"] == "assistant"):
                dropped["unsafe_assistant"] += 1
                continue
            if any(DIAGNOSIS.search(m["content"]) for m in messages if m["role"] == "assistant"):
                dropped["diagnosis_claim"] += 1
                continue
            key = hashlib.md5(
                (messages[0]["content"] + "|" + "".join(m["content"] for m in messages)).encode()
            ).hexdigest()
            if key in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(key)
            record["messages"] = messages
            sft_records.append(record)

    pair_records, pair_seen, pair_dropped = [], set(), Counter()
    for raw in args.pairs:
        path = Path(raw)
        if not path.exists():
            continue
        for line in path.open("r", encoding="utf-8"):
            if not line.strip():
                continue
            record = json.loads(line)
            for field in ("prompt", "chosen", "rejected"):
                if not record.get(field):
                    pair_dropped[f"missing_{field}"] += 1
                    record = None
                    break
            if record is None:
                continue
            record["prompt"] = redact_pii(record["prompt"])
            record["chosen"] = redact_pii(record["chosen"])
            record["rejected"] = redact_pii(record["rejected"])
            if unsafe_text(record["chosen"]) or unsafe_text(record["rejected"]):
                pair_dropped["unsafe"] += 1
                continue
            key = hashlib.md5((record["prompt"] + record["chosen"]).encode()).hexdigest()
            if key in pair_seen:
                pair_dropped["duplicate"] += 1
                continue
            pair_seen.add(key)
            pair_records.append(record)

    sft_path = Path(args.sft_out)
    sft_path.parent.mkdir(parents=True, exist_ok=True)
    sft_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in sft_records) + "\n", encoding="utf-8"
    )
    pairs_path = Path(args.pairs_out)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in pair_records) + "\n", encoding="utf-8"
    )
    stats = {
        "sft": {"kept": len(sft_records), "dropped": dict(dropped)},
        "preference": {"kept": len(pair_records), "dropped": dict(pair_dropped)},
        "outputs": [str(sft_path), str(pairs_path)],
    }
    (ROOT / "reports" / "data_build.json").parent.mkdir(exist_ok=True)
    (ROOT / "reports" / "data_build.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
