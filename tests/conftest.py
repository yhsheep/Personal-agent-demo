"""测试环境：强制 SQLite + 本地向量 + Mock LLM，避免外部依赖。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TEST_DIR = Path(tempfile.mkdtemp(prefix="wenlan-tests-"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_DIR / 'test.db').as_posix()}"
os.environ["USE_EMBEDDED_POSTGRES"] = "false"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["EMBEDDING_DIM"] = "256"
os.environ["LOG_LEVEL"] = "WARNING"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db import init_db  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    init_db()
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> Path:
    from app.services.pdf_parser import create_pdf_from_text

    text = (
        "Abstract\n"
        "This paper studies retrieval augmented generation with dense vectors and a language model. "
        "The retriever uses DPR and the generator uses BART. "
        + "Retrieval augmented generation improves factual question answering. " * 18
        + "\n\n1 Introduction\n"
        + "We describe the method and the experiments in detail. " * 24
        + "\n\n2 Method\n"
        + "The generator is BART and the retriever is DPR. " * 28
        + "\n\n3 Experiments\n"
        + "We evaluate on Natural Questions and TriviaQA. " * 20
    )
    path = tmp_path_factory.mktemp("pdf") / "sample.pdf"
    create_pdf_from_text(
        text,
        path,
        title="A Test Paper on Retrieval Augmented Generation",
        authors="Alice Zhang, Bob Li",
    )
    return path


@pytest.fixture(scope="session")
def uploaded_paper(client, sample_pdf):
    with sample_pdf.open("rb") as fh:
        resp = client.post(
            "/api/papers/upload",
            files={"file": ("sample.pdf", fh, "application/pdf")},
            data={"title": "A Test Paper on Retrieval Augmented Generation", "authors": "Alice Zhang, Bob Li"},
        )
    assert resp.status_code == 200, resp.text
    return resp.json()["paper"]
