FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 运行依赖（PyMuPDF / psycopg 均提供 wheel；无需编译工具）
COPY pyproject.toml README.md ./
COPY app ./app
COPY data/seed ./data/seed
COPY data/eval ./data/eval

RUN pip install --upgrade pip && pip install . \
    && mkdir -p data/uploads data/logs

ENV USE_EMBEDDED_POSTGRES=false \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
