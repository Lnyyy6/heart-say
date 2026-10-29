import json
from pathlib import Path

from heart_say.rag.knowledge import load_chunks, load_records
from heart_say.schemas import EvalCase

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def test_knowledge_base_schema_and_count():
    records = load_records()
    chunks = load_chunks()
    assert len(records) == 80
    assert chunks
    assert all(record["id"] and record["text"] for record in records)
    assert all(record["source_name"] for record in records)


def test_sft_and_preference_counts():
    sft = read_jsonl(ROOT / "data/sft/seed_sft.jsonl")
    pairs = read_jsonl(ROOT / "data/preference/seed_pairs.jsonl")
    assert len(sft) == 60
    assert len(pairs) == 30
    for record in sft:
        roles = [message["role"] for message in record["messages"]]
        assert roles[0] == "user"
        assert roles[-1] == "assistant"
        assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))
    for record in pairs:
        assert record["chosen"] != record["rejected"]


def test_eval_and_redteam_schema():
    cases = [EvalCase(**row) for row in read_jsonl(ROOT / "data/eval/cases.jsonl")]
    redteam = read_jsonl(ROOT / "data/redteam/prompts.jsonl")
    assert len(cases) == 100
    assert len(redteam) == 50
    assert sum(1 for case in cases if case.expected_risk in ("high", "critical")) >= 10
    crisis_redteam = [item for item in redteam if item["category"] == "crisis"]
    assert crisis_redteam
    assert all("12356" in item["expected_behavior"] for item in crisis_redteam)
