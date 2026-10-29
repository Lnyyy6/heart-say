from fastapi.testclient import TestClient

from app.api.main import app


def test_health_and_meta():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        meta = client.get("/meta")
        assert meta.status_code == 200
        assert "12356" in meta.json()["disclaimer"]


def test_chat_endpoint():
    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "孩子最近总说不想上学，我该怎么开口？",
                "session_id": "apitest",
                "use_rag": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "apitest"
        assert len(data["text"]) > 20
        assert data["safety"]["disclaimer"]
        assert data["trace"]["use_rag"] is True


def test_session_endpoint():
    with TestClient(app) as client:
        client.post(
            "/chat",
            json={"message": "我最近很焦虑，睡不着。", "session_id": "apitest2"},
        )
        response = client.get("/sessions/apitest2")
        assert response.status_code == 200
        assert len(response.json()["messages"]) >= 2
