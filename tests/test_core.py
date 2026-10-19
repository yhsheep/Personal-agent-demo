from app.db import init_db, session_scope
from app.models import Paper
from app.services.chunker import chunk_pages
from app.services.embeddings import LocalHashEmbedding
from app.utils.text import expand_concepts, normalize_text, sanitize_text, sanitize_value


def test_chunking_is_section_aware_and_bounded():
    pages = [
        {
            "page": 1,
            "text": (
                "Abstract\n\n"
                "This paper studies retrieval augmented generation. " * 20
                + "\n\n1 Introduction\n\n"
                + "Large language models store knowledge in parameters. " * 25
                + "\n\n2 Method\n\n"
                + "The retriever is DPR and the generator is BART. " * 20
            ),
        }
    ]
    chunks = chunk_pages(pages, chunk_size=320, overlap=50)
    assert len(chunks) >= 3
    assert all(c.char_count <= 420 for c in chunks)
    assert any("Method" in c.section for c in chunks)
    assert any("BART" in c.content for c in chunks)
    # 相邻块有重叠
    assert chunks[0].content[-30:] in chunks[1].content or chunks[1].content[:30] in chunks[0].content


def test_local_embedding_normalized_and_bilingual():
    emb = LocalHashEmbedding(256)
    a = emb.embed_one("RAG 的生成器是什么？")
    b = emb.embed_one("The generator is BART and the retriever is DPR.")
    same = emb.embed_one("RAG 的生成器是什么？")
    assert len(a) == 256
    assert abs(sum(x * x for x in a) ** 0.5 - 1.0) < 1e-4
    assert sum(x * y for x, y in zip(a, same, strict=False)) > 0.999
    # 中英术语桥接应产生共享特征，而不是完全正交。
    assert sum(x * y for x, y in zip(a, b, strict=False)) != 0.0


def test_glossary_expands_specific_terms():
    terms = expand_concepts("生成器使用什么模型")
    assert "generator" in terms
    assert "model" in terms


def test_nul_and_control_characters_are_sanitized():
    assert sanitize_text("a\x00b\x01c") == "ab c"
    assert "\x00" not in normalize_text("title\x00with\x00nul\nnext")
    nested = sanitize_value({"a": ["x\x00y", {"b": "p\x00q"}]})
    assert nested == {"a": ["xy", {"b": "pq"}]}


def test_orm_flush_never_writes_nul_bytes():
    init_db()
    with session_scope() as session:
        paper = Paper(
            title="NUL test",
            authors=["A\x00uthor"],
            abstract="abstract\x00text",
            raw_text="raw\x00text with \x01 control",
            page_texts=[{"page": 1, "text": "page\x00text"}],
            keywords=["key\x00word"],
            source="test",
            parse_status="parsed",
        )
        session.add(paper)
        session.flush()
        paper_id = paper.id
    with session_scope() as session:
        stored = session.get(Paper, paper_id)
        assert stored is not None
        assert "\x00" not in stored.raw_text
        assert "\x00" not in stored.abstract
        assert "\x00" not in stored.page_texts[0]["text"]
        assert all("\x00" not in author for author in stored.authors)
        session.delete(stored)

