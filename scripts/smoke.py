"""端到端冒烟测试：健康检查 → RAG → 问答 → 清理会话。

用法：python scripts/smoke.py
"""

from __future__ import annotations

import sys

import httpx

BASE = "http://127.0.0.1:8000"


def main() -> int:
    with httpx.Client(timeout=60.0) as client:
        health = client.get(f"{BASE}/api/health").json()
        print("health:", health["status"], health["database"]["dialect"])

        stats = client.get(f"{BASE}/api/library/stats").json()
        print("library:", stats["paper_count"], "papers,", stats["chunk_count"], "chunks")

        rag = client.post(
            f"{BASE}/api/rag/search",
            json={"query": "RAG 的生成器是什么模型？", "top_k": 4},
        )
        rag.raise_for_status()
        results = rag.json()["results"]
        print("rag:", len(results), "hits;", results[0]["paper_title"][:60] if results else "none")

        chat = client.post(
            f"{BASE}/api/chat",
            json={"message": "RAG 的生成器是什么模型？", "top_k": 6},
        )
        chat.raise_for_status()
        data = chat.json()
        print("chat:", data["message"]["content"][:180].replace("\n", " "))
        print("citations:", len(data["citations"]), "trace:", [t["name"] for t in data["tool_trace"]])
        client.delete(f"{BASE}/api/conversations/{data['conversation_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
