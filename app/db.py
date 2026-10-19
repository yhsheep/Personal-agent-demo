"""数据库连接：默认启动 pip 内置的 PostgreSQL 16 + pgvector。

也支持外部 PostgreSQL 与 SQLite 测试降级。向量列在 PostgreSQL 上是原生
``vector(1024)``，在 SQLite 上退化为 JSON，由应用层做余弦相似度计算。
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.models import Base
from app.utils.text import sanitize_value

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None
_pg_server: Any = None
_database_url: str | None = None
_lock = threading.Lock()


@event.listens_for(Session, "before_flush")
def _sanitize_text_fields(session: Session, _flush_context, _instances) -> None:
    """数据库写入兜底：清除 PostgreSQL 不接受的 NUL / 控制字符。

    PDF 解析器、外部 API、用户输入都可能夹带 NUL。这里统一在 ORM flush
    前递归清理字符串与 JSON，避免任何入口把 0x00 写进 text/json 列。
    """
    for obj in list(session.new) + list(session.dirty):
        mapper = getattr(obj, "__mapper__", None)
        if mapper is None:
            continue
        for attr in mapper.column_attrs:
            value = getattr(obj, attr.key, None)
            cleaned = sanitize_value(value)
            if cleaned != value:
                setattr(obj, attr.key, cleaned)


def _normalize_postgres_url(url: str) -> str:
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


def _running_pg_from_pidfile() -> tuple[str, int] | None:
    """从 postmaster.pid 读取仍然存活的本机 PostgreSQL 端口。"""
    pid_file = settings.pgdata_path / "postmaster.pid"
    if not pid_file.exists():
        return None
    try:
        lines = pid_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        pid = int(lines[0].strip())
        port = int(lines[3].strip())
    except (ValueError, IndexError):
        return None
    try:
        import psutil

        proc = psutil.Process(pid)
        if "postgres" not in (proc.name() or "").lower():
            return None
    except Exception:
        try:
            import psutil

            if not psutil.pid_exists(pid):
                return None
        except Exception:
            pass
    sock = socket.socket()
    sock.settimeout(1.5)
    try:
        sock.connect(("127.0.0.1", port))
        return "127.0.0.1", port
    except OSError:
        return None
    finally:
        sock.close()


def _wait_for_recovered_postgres(attempts: int = 20, delay: float = 2.0) -> str | None:
    """pg_ctl 超时但 PostgreSQL 可能仍在做崩溃恢复。

    等待 postmaster.pid 对应端口可连接，并用 trust 认证试连，成功后
    直接返回 SQLAlchemy URL，避免把已有论文误降级到空 SQLite。
    """
    pid_file = settings.pgdata_path / "postmaster.pid"
    if not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text(encoding="utf-8", errors="ignore").splitlines()[0])
        import psutil

        if not psutil.pid_exists(pid):
            return None
    except Exception:
        return None
    for index in range(attempts):
        endpoint = _running_pg_from_pidfile()
        if endpoint:
            host, port = endpoint
            url = f"postgresql+psycopg://postgres@{host}:{port}/postgres"
            try:
                import psycopg

                with psycopg.connect(
                    f"postgresql://postgres@{host}:{port}/postgres",
                    connect_timeout=2,
                ):
                    pass
                logger.info(
                    "检测到已恢复的嵌入式 PostgreSQL（端口 %s），直接连接", port
                )
                return url
            except Exception as exc:
                logger.debug("PostgreSQL 端口已开但暂未接受连接：%s", exc)
        if index < attempts - 1:
            time.sleep(delay)
    return None


def _start_embedded_postgres() -> str | None:
    global _pg_server
    try:
        import pgserver  # type: ignore
    except Exception as exc:  # pragma: no cover - 取决于安装环境
        logger.warning("pgserver 不可用，将尝试 SQLite 降级：%s", exc)
        return None

    settings.pgdata_path.mkdir(parents=True, exist_ok=True)
    logger.info("正在启动嵌入式 PostgreSQL，数据目录：%s", settings.pgdata_path)
    try:
        _pg_server = pgserver.get_server(str(settings.pgdata_path), cleanup_mode="stop")
        _pg_server.psql("CREATE EXTENSION IF NOT EXISTS vector")
        uri = _pg_server.get_uri()
        logger.info("嵌入式 PostgreSQL 已启动（pgvector enabled）")
        return _normalize_postgres_url(uri)
    except Exception as exc:
        logger.exception("嵌入式 PostgreSQL 启动失败：%s", exc)
        # Windows 上 pg_ctl -w 的 10 秒超时可能早于崩溃恢复完成。
        # 若 postmaster.pid 指向的进程已经起来，等待并直接连接，避免
        # 静默回退到空 SQLite 导致用户误以为文献库丢失。
        recovered = _wait_for_recovered_postgres()
        if recovered:
            return recovered
        return None


def resolve_database_url() -> str:
    if settings.database_url:
        return _normalize_postgres_url(settings.database_url)
    if settings.use_embedded_postgres:
        url = _start_embedded_postgres()
        if url:
            return url
        if not settings.sqlite_fallback:
            raise RuntimeError("嵌入式 PostgreSQL 启动失败，且 SQLITE_FALLBACK=false")
    db_path = (settings.data_dir / "wenlan.db").as_posix()
    logger.warning("使用 SQLite 降级数据库：%s（仅推荐测试）", db_path)
    return f"sqlite:///{db_path}"


def get_engine() -> Engine:
    global _engine, _database_url
    if _engine is not None:
        return _engine
    with _lock:
        if _engine is not None:
            return _engine
        _database_url = resolve_database_url()
        connect_args: dict[str, Any] = {}
        if _database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        if _database_url.startswith("sqlite"):
            engine = create_engine(
                _database_url,
                connect_args=connect_args,
                pool_pre_ping=True,
                future=True,
            )

            @event.listens_for(engine, "connect")
            def _sqlite_pragma(dbapi_conn, _):  # type: ignore[no-untyped-def]
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        else:
            engine = create_engine(
                _database_url,
                connect_args=connect_args,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
                future=True,
            )
        _engine = engine
        return engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(), autoflush=False, autocommit=False, expire_on_commit=False
        )
    return _session_factory


def get_db() -> Generator[Session, None, None]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """创建扩展、表与向量索引。可重复调用。"""
    engine = get_engine()
    dialect = engine.dialect.name
    if dialect == "postgresql":
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    if dialect == "postgresql":
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_rag_chunks_embedding_hnsw "
                    "ON rag_chunks USING hnsw (embedding vector_cosine_ops)"
                )
            )
            conn.execute(text("ANALYZE papers"))
            conn.execute(text("ANALYZE rag_chunks"))


def db_status() -> dict[str, Any]:
    engine = get_engine()
    dialect = engine.dialect.name
    info: dict[str, Any] = {"dialect": dialect, "embedded": _pg_server is not None}
    if _database_url:
        try:
            parsed = make_url(_database_url)
            info["database"] = parsed.database
            info["host"] = parsed.host
            info["port"] = parsed.port
        except Exception:
            pass
    try:
        with engine.connect() as conn:
            if dialect == "postgresql":
                version = conn.execute(text("SHOW server_version")).scalar_one()
                ext = conn.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                ).scalar_one_or_none()
                info["server_version"] = version
                info["pgvector_version"] = ext
            else:
                info["server_version"] = conn.execute(
                    text("SELECT sqlite_version()")
                ).scalar_one()
        info["ok"] = True
    except Exception as exc:
        info["ok"] = False
        info["error"] = str(exc)
    return info


def reset_database() -> None:
    """危险操作：删除全部表后重建（CLI 使用）。"""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    init_db()
