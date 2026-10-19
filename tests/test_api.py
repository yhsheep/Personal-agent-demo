import json


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["demo_mode"] is True
    assert data["database"]["dialect"] == "sqlite"


def test_library_upload_and_rag(client, uploaded_paper):
    assert uploaded_paper["chunk_count"] > 0
    resp = client.get("/api/papers")
    assert resp.status_code == 200
    assert resp.json()["total"] >= 1
    pid = uploaded_paper["id"]
    detail = client.get(f"/api/papers/{pid}")
    assert detail.status_code == 200
    assert detail.json()["title"].startswith("A Test Paper")
    chunks = client.get(f"/api/papers/{pid}/chunks")
    assert chunks.status_code == 200
    assert chunks.json()["total"] == uploaded_paper["chunk_count"]

    rag = client.post("/api/rag/search", json={"query": "generator BART", "top_k": 3})
    assert rag.status_code == 200
    data = rag.json()
    assert data["count"] >= 1
    assert any("BART" in r["content"] for r in data["results"])


def test_chat_stream_and_history(client, uploaded_paper):
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"message": "这篇论文的生成器使用什么模型？", "top_k": 4},
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    assert "event: token" in body
    assert "event: done" in body
    events = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        lines = block.splitlines()
        event = next((line[6:].strip() for line in lines if line.startswith("event:")), "")
        data = next((line[5:].strip() for line in lines if line.startswith("data:")), "{}")
        if event:
            events.append((event, json.loads(data)))
    done = next(data for event, data in events if event == "done")
    conv_id = done["conversation_id"]
    assert done["message_id"]
    history = client.get(f"/api/conversations/{conv_id}")
    assert history.status_code == 200
    messages = history.json()["messages"]
    assert messages[0]["role"] == "user"
    assert messages[-1]["role"] == "assistant"
    assert len(messages[-1]["citations"]) >= 1


def test_evaluation_run(client, uploaded_paper):
    resp = client.post(
        "/api/evaluations/run",
        json={"kind": "retrieval", "limit": 2, "top_k": 3, "name": "pytest retrieval"},
    )
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run"]["id"]
    import time

    status = None
    for _ in range(40):
        detail = client.get(f"/api/evaluations/runs/{run_id}")
        assert detail.status_code == 200
        status = detail.json()["run"]["status"]
        if status not in {"queued", "running"}:
            break
        time.sleep(0.25)
    assert status == "completed"
    metrics = detail.json()["run"]["metrics"]
    assert "retrieval_recall_at_k" in metrics
    assert metrics["case_count"] == 2


def test_conversation_lifecycle(client):
    created = client.post("/api/conversations", json={"title": "测试会话"})
    assert created.status_code == 200
    cid = created.json()["conversation"]["id"]
    renamed = client.patch(f"/api/conversations/{cid}", json={"title": "改名后的会话"})
    assert renamed.status_code == 200
    assert renamed.json()["conversation"]["title"] == "改名后的会话"
    deleted = client.delete(f"/api/conversations/{cid}")
    assert deleted.status_code == 200
