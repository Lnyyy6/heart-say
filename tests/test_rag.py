import numpy as np

from heart_say.rag.embedder import HashEmbedder
from heart_say.rag.retriever import get_retriever
from heart_say.rag.text import chunk_record, tokenize


def test_tokenize_chinese():
    tokens = tokenize("孩子中考前失眠，家长也很焦虑")
    assert "失眠" in tokens or "睡不着" in tokens
    assert "焦虑" in tokens


def test_chunk_record_within_limit():
    record = {
        "id": "test-1",
        "title": "测试",
        "text": "第一句。" * 80,
        "topic": "test",
    }
    chunks = chunk_record(record, max_chars=200)
    assert len(chunks) > 1
    assert all(len(chunk["text"]) <= 260 for chunk in chunks)
    assert chunks[0]["doc_id"] == "test-1"


def test_hash_embedder_deterministic_and_normalized():
    embedder = HashEmbedder(dim=128)
    first = embedder.encode(["孩子考试焦虑"])
    second = embedder.encode(["孩子考试焦虑"])
    assert first.shape == (1, 128)
    assert np.allclose(first, second)
    assert abs(float(np.linalg.norm(first[0])) - 1.0) < 1e-5


def test_retrieval_prefers_relevant_topic():
    retriever = get_retriever()
    docs = retriever.retrieve("孩子初三中考前天天失眠，我比他还焦虑")
    assert docs
    assert any(doc.topic in ("sleep", "study") for doc in docs[:3])
    assert all(doc.topic != "crisis" for doc in docs)


def test_retrieval_filters_parent_docs_for_teen():
    retriever = get_retriever()
    docs = retriever.retrieve("我一考试就睡不着，总怕考砸", audience="teen")
    assert docs
    assert all(doc.audience != "parent" for doc in docs)
