"""FastAPI 应用入口。"""

from __future__ import annotations

import logging
import logging.handlers
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import chat, evaluations, papers, system
from app.config import settings
from app.db import db_status, init_db
from app.services.embeddings import get_embedding_service
from app.services.llm import get_llm_service

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"


def configure_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        root.addHandler(stream)
    logfile = settings.logs_dir / "wenlan.log"
    if not any(
        isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers
    ):
        fh = logging.handlers.RotatingFileHandler(
            logfile, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setFormatter(formatter)
        root.addHandler(fh)
    logging.getLogger("httpx").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    started = time.perf_counter()
    logger.info("启动 %s v%s", settings.app_name, __version__)
    init_db()
    db = db_status()
    logger.info("数据库：%s", db)
    logger.info("LLM：%s", get_llm_service().status())
    logger.info("Embedding：%s", get_embedding_service().status())
    logger.info("启动完成，用时 %.1fs", time.perf_counter() - started)
    yield
    logger.info("应用关闭")


app = FastAPI(
    title="文澜 WenLan · 文献问答与 RAG 评测",
    description=(
        "基于 FastAPI、Tool Calling 与 pgvector RAG 的本地文献问答项目："
        "PDF 知识库、多轮问答、引用溯源、论文检索与评测。"
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router, prefix="/api")
app.include_router(papers.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(evaluations.router, prefix="/api")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    file = STATIC_DIR / "index.html"
    if not file.exists():
        return JSONResponse(
            {
                "message": "文澜 API 已启动，但前端尚未构建。",
                "docs": "/api/docs",
            }
        )
    return FileResponse(file, media_type="text/html")


@app.get("/favicon.svg", include_in_schema=False)
def favicon():
    file = STATIC_DIR / "favicon.svg"
    if file.exists():
        return FileResponse(file, media_type="image/svg+xml")
    return JSONResponse({"message": "not found"}, status_code=404)


@app.get("/api", include_in_schema=False)
def api_root() -> dict:
    return {
        "name": settings.app_name,
        "version": __version__,
        "docs": "/api/docs",
        "health": "/api/health",
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("未处理异常：%s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务器内部错误：{exc}"},
    )
