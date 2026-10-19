"""命令行工具：wenlan init-db / seed / serve / eval / experiments / status。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from app import __version__
from app.config import settings
from app.db import db_status, init_db, reset_database, session_scope
from app.models import EvaluationRun
from app.services.evaluation import get_evaluation_service, load_dataset, run_to_dict
from app.services.seed import seed_demo


def cmd_init_db(_: argparse.Namespace) -> int:
    init_db()
    print(json.dumps(db_status(), ensure_ascii=False, indent=2))
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    from sqlalchemy import func, select

    from app.models import Chunk, Conversation, Paper
    from app.services.embeddings import get_embedding_service
    from app.services.llm import get_llm_service

    init_db()
    with session_scope() as session:
        print(f"文澜 WenLan v{__version__}")
        print(json.dumps(db_status(), ensure_ascii=False, indent=2))
        print("LLM:", json.dumps(get_llm_service().status(), ensure_ascii=False))
        print("Embedding:", json.dumps(get_embedding_service().status(), ensure_ascii=False))
        for name, model in (
            ("papers", Paper),
            ("chunks", Chunk),
            ("conversations", Conversation),
        ):
            count = session.scalar(select(func.count(model.id))) or 0
            print(f"{name}: {count}")
    try:
        cases = load_dataset()
        print(f"eval_cases: {len(cases)}")
    except Exception as exc:
        print(f"eval_cases: error {exc}")
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    init_db()
    with session_scope() as session:
        result = seed_demo(session, download=args.download, force=args.force)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
    return 0


def _create_run(session, kind: str, limit: int | None, top_k: int, prompt_variant: str) -> str:
    run = EvaluationRun(
        name=f"CLI {kind} {datetime.now().strftime('%m-%d %H:%M')}",
        kind=kind,
        status="queued",
        dataset=settings.eval_dataset_path.name,
        config={
            "limit": limit,
            "top_k": top_k,
            "prompt_variant": prompt_variant,
            "chunk_sizes": [300, 500, 700],
            "overlaps": [50, 100],
            "top_ks": [3, 5, 8],
        },
        llm_provider="",
        embedding_model="",
    )
    session.add(run)
    session.commit()
    return run.id


def cmd_eval(args: argparse.Namespace) -> int:
    init_db()
    service = get_evaluation_service()
    kind = "full" if args.kind == "full" else args.kind
    with session_scope() as session:
        run_id = _create_run(session, kind, args.limit, args.top_k, args.prompt)
        run = session.get(EvaluationRun, run_id)
        service.execute_run(session, run_id, kind, run.config)
        session.refresh(run)
        print(json.dumps(run_to_dict(run, include_details=False), ensure_ascii=False, indent=2))
    return 0


def cmd_experiments(args: argparse.Namespace) -> int:
    init_db()
    service = get_evaluation_service()
    kind = args.kind
    with session_scope() as session:
        run_id = _create_run(session, kind, args.limit, args.top_k, "citation_strict")
        run = session.get(EvaluationRun, run_id)
        service.execute_run(session, run_id, kind, run.config)
        session.refresh(run)
        print(json.dumps(run_to_dict(run, include_details=False), ensure_ascii=False, indent=2))
    return 0


def cmd_reset_db(args: argparse.Namespace) -> int:
    if not args.yes:
        print("危险操作：将删除所有表。请加 --yes 确认。", file=sys.stderr)
        return 2
    reset_database()
    print(json.dumps(db_status(), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wenlan", description="文澜 WenLan 文献问答与 RAG 评测"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-db", help="初始化数据库与 pgvector 扩展")
    p.set_defaults(func=cmd_init_db)

    p = sub.add_parser("status", help="查看系统状态与数据统计")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("seed", help="导入示例论文库")
    p.add_argument("--download", action="store_true", help="尝试下载真实 arXiv PDF")
    p.add_argument("--force", action="store_true", help="保留参数，当前幂等导入")
    p.set_defaults(func=cmd_seed)

    p = sub.add_parser("serve", help="启动 Web 服务")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("eval", help="运行 Agent 评测（同步）")
    p.add_argument("--kind", choices=["full", "retrieval", "agent"], default="full")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--prompt", default="citation_strict")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("experiments", help="运行 Chunk/Top-K 或 Prompt 参数实验")
    p.add_argument("--kind", choices=["retrieval_grid", "prompt_grid"], default="retrieval_grid")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--top-k", type=int, default=6)
    p.set_defaults(func=cmd_experiments)

    p = sub.add_parser("reset-db", help="删除并重建所有表（危险）")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_reset_db)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
