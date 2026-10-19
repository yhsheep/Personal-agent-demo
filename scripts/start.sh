#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "文澜 WenLan | 文献问答与 RAG 评测"

if [ ! -d ".venv" ]; then
  echo "[1/5] 创建虚拟环境 .venv"
  python3 -m venv .venv
fi
source .venv/bin/activate
PY=python

echo "[2/5] 安装依赖"
python -m pip install --upgrade pip >/dev/null
python -m pip install -e ".[dev]"

if [ ! -f ".env" ]; then
  echo "[3/5] 创建 .env"
  cp .env.example .env
else
  echo "[3/5] 已存在 .env"
fi

echo "[4/5] 初始化数据库与 pgvector"
python -m app.cli init-db

echo "[5/5] 数据库已就绪；示例库不会自动导入，可在页面中手动选择。"

echo "启动： http://127.0.0.1:8000"
exec python -m app.cli serve
