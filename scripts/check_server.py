"""检查本地服务是否可访问，并输出诊断信息。"""

from __future__ import annotations

import json
import re
import socket
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "logs" / "check_server.json"


def main() -> None:
    result: dict = {}
    sock = socket.socket()
    sock.settimeout(2)
    try:
        sock.connect(("127.0.0.1", 8000))
        result["tcp_8000"] = "open"
    except Exception as exc:
        result["tcp_8000"] = f"closed: {exc}"
    finally:
        sock.close()
    try:
        resp = httpx.get("http://127.0.0.1:8000/api/health", timeout=5)
        result["http_status"] = resp.status_code
        result["health"] = resp.json()
        home = httpx.get("http://127.0.0.1:8000/", timeout=5)
        match = re.search(r"<title>(.*?)</title>", home.text, re.S)
        result["page_title"] = match.group(1).strip() if match else "no title"
        result["page_has_new_copy"] = "文献问答与 RAG 评测" in home.text
    except Exception as exc:
        result["http_error"] = str(exc)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT.write_text(json.dumps({"fatal": str(exc)}, ensure_ascii=False), encoding="utf-8")
        print(exc, file=sys.stderr)
        raise
