# 架构与数据模型

## 1. 总览

文澜是一个单体 FastAPI 应用：同一个进程同时提供 REST API、SSE 流与静态 Web UI。运行形态刻意保持简单——不依赖 Node、Redis、Celery 或 Docker，下载 Python 依赖即可启动。

```text
Browser
  │  HTML/CSS/JS（零构建）
  ▼
FastAPI (sync endpoints + StreamingResponse)
  ├─ AgentService ── LLMService（OpenAI-compatible / Mock）
  │      └─ ToolRegistry ── PaperSearch / RAG / Library
  ├─ LibraryService ── PDFParser → Chunker → Embedding → pgvector
  ├─ EvaluationService ── Retrieval / Agent / Grid / Prompt
  └─ SQLAlchemy Session ── PostgreSQL 16 + pgvector
```

所有业务路由使用同步 `def`，由 Starlette 放入线程池执行；这样可以直接使用同步 SQLAlchemy、`httpx.Client` 与 PyMuPDF，避免异步生态的版本兼容问题。SSE 通过同步生成器返回 `text/event-stream`。

## 2. 数据库

项目支持两种后端：

| 后端 | 用途 | 向量 |
| --- | --- | --- |
| PostgreSQL 16 + pgvector | 默认。`pgserver` 随 pip 安装并启动；也支持外部实例 | 原生 `vector(1024)` + HNSW Cosine |
| SQLite | 单元测试 / 极端降级 | JSON 存向量，应用层 numpy 余弦 |

`app/db.py` 的解析顺序：

1. `DATABASE_URL`（外部 PostgreSQL 优先）
2. `USE_EMBEDDED_POSTGRES=true` → `pgserver.get_server(data/pgdata)`，随机端口，自动 `CREATE EXTENSION vector`
3. `SQLITE_FALLBACK=true` → `data/wenlan.db`

`init_db()` 幂等执行：

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS ...;  -- SQLAlchemy metadata.create_all
CREATE INDEX IF NOT EXISTS ix_rag_chunks_embedding_hnsw
  ON rag_chunks USING hnsw (embedding vector_cosine_ops);
ANALYZE papers;
ANALYZE rag_chunks;
```

### 表

#### `papers`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | varchar(32) | UUID hex |
| `title` / `authors` / `abstract` | text/json | 元数据 |
| `year` / `venue` / `doi` / `arxiv_id` / `url` | — | 来源 |
| `raw_text` / `page_texts` | text/json | 原文与页级文本，用于重建索引 |
| `pdf_path` / `pdf_sha256` / `file_size` | — | 本地 PDF 与去重 |
| `page_count` / `chunk_count` | int | 统计 |
| `parse_status` | varchar | `metadata` / `parsed` / `indexed` / `parse_failed` |
| `keywords` | json | 主题别名 + 自动关键词 |
| `indexed_at` / `created_at` / `updated_at` | timestamptz | — |

#### `rag_chunks`

| 字段 | 说明 |
| --- | --- |
| `paper_id` | 外键，级联删除 |
| `content` | 原始文本块（回答引用展示） |
| `section` | 识别到的章节名 |
| `page_start` / `page_end` | 引用页码 |
| `chunk_index` | 论文内顺序 |
| `token_count` / `char_count` | 统计 |
| `embedding` | `vector(1024)`，L2 归一化 |
| `embedding_model` | 记录模型，便于判断是否需要重建 |

#### `conversations` / `messages`

消息保存 `role`、`content`、`citations`（含 `cited` 标记）、`tool_trace`、`usage`、`latency_ms`。  
多轮上下文只回放 user/assistant 的最终文本，不回放历史 Tool Message，避免上下文膨胀与跨提供商格式差异。

#### `evaluation_runs`

一次评测/实验的完整快照：`kind`、`status`、`config`、`metrics`、`details`、`llm_provider`、`embedding_model`。  
后台线程执行，前端轮询 `GET /api/evaluations/runs/{id}`；状态：`queued → running → completed | failed`。

## 3. PDF 与分块

`pdf_parser.py`：

- 校验加密、页数、可提取文本量（扫描版会提示 OCR）；
- 用 block + sort 提取页面文本；
- 清理重复页眉/页脚（出现于 ≥40% 页面）；
- 首页最大字号 + PDF metadata 推断标题；
- 正则抽取 `Abstract` / `摘要`。

`chunker.py`：

1. 页面文本按空行拆 block；
2. 识别 `1 Introduction` / `Method` / `实验` 等标题，更新 `section`；
3. 句子切分，超长句按逗号/冒号二次切分；
4. 按 `chunk_size` 累积，优先在章节边界 flush；
5. `overlap` 以尾部句子回填，避免硬切；
6. 去重、最短长度过滤、重新编号。

默认 `CHUNK_SIZE=600`、`CHUNK_OVERLAP=100`。可通过网格实验调整。

## 4. Embedding 与索引

`EmbeddingService` 统一两种实现：

- **远端**：OpenAI 兼容 `/embeddings`，支持 SiliconFlow `BAAI/bge-m3`、DashScope `text-embedding-v4`、OpenAI。
- **本地**：signed feature hashing（1024 维），中英文词、中文 2/3-gram、术语桥接；无需下载模型，保证离线可运行。

远端向量会做 L2 归一化并校验维度；维度不匹配时直接报错，避免写入不可检索的向量。

入库时 `RAGService.index_paper()`：

1. `chunk_pages` 生成文本块；
2. Embedding（批量）；
3. 删除旧 chunk，写入新 chunk；
4. 更新 `papers.chunk_count` / `indexed_at` / `parse_status`。

## 5. 检索与重排

```text
query
  ├─ query embedding
  ├─ pgvector: 1 - (embedding <=> qvec)  [Top 5K 候选]
  ├─ 词法: IDF 加权 term overlap（论文标题 + 主题别名 + 正文）
  ├─ 融合: local embedding → 0.45 vector + 0.55 lexical + 0.10 title
  │         remote embedding → 0.82 vector + 0.18 lexical + 0.04 title
  ├─ min_score 过滤
  ├─ 每篇论文最多 3 个 chunk（多样性）
  └─ Top-K
```

要点：

- 向量仅编码正文/章节，不把标题/别名写进向量，避免“作者/结论页”因标题重复而抢占排名；
- 标题与主题别名参与重排，解决“原始 RAG / DPR / GraphRAG”等中文指代；
- IDF 在候选集上近似计算，让 `limitation`、`external`、`MaxSim` 等稀有意图词权重更高；
- 每篇论文配额防止单个长文档占满上下文。

pgvector SQL：

```sql
SELECT ..., 1 - (c.embedding <=> CAST(:qvec AS vector)) AS vector_score
FROM rag_chunks c JOIN papers p ON p.id = c.paper_id
ORDER BY c.embedding <=> CAST(:qvec AS vector)
LIMIT :limit;
```

`CAST(:qvec AS vector)` 传入 `[0.1,0.2,...]` 字符串，避免 psycopg 对数组类型的适配差异。

## 6. Agent 循环

`AgentService` 每轮：

1. 组装 system prompt（Prompt 变体 + 当前日期 + 引用规则）；
2. 回放最近 `AGENT_HISTORY_TURNS` 轮 user/assistant；
3. `LLM.complete(messages, tools=schemas)`；
4. 无 tool_calls → 模型直接回答；
5. 有 tool_calls → 记录 assistant tool_calls，逐个执行，写回 `role=tool`；
6. 最多 `AGENT_MAX_TOOL_ROUNDS=4` 轮；
7. 超出上限 → 强制一次无工具收束。

引用编号由 `ToolContext` 统一分配：同一 chunk 多次检索复用同一编号，不同检索递增，避免 `[1]` 冲突。

工具执行异常会作为 `{"ok": false, "error": ...}` 回传给模型，让模型有机会换关键词重试，而不是让整个对话失败。

## 7. 流式协议

`POST /api/chat/stream` 返回：

```text
event: start
data: {"conversation_id": "...", "llm": {...}}

event: tool_start
data: {"name": "search_library", "arguments": {...}}

event: tool_result
data: {"name": "search_library", "status": "success", "latency_ms": 12, "summary": "检索到 8 个原文片段"}

event: token
data: {"text": "根据..."}

event: citations
data: {"citations": [...]}

event: done
data: {"message_id": "...", "tool_trace": [...], "usage": {...}, "latency_ms": 38}
```

前端逐块解析并渲染 Markdown；滚动只在用户接近底部时自动跟随。

## 8. 评测设计

- **检索**：Recall@K（论文级）、MRR、nDCG@K、Precision@K、hit rate；
- **Agent**：必需工具覆盖率（Tool Calling Accuracy）、exact match、tool F1；
- **任务**：关键词事实组 + 引用落地 + 拒答/错误校验；
- **实验**：Chunk × Overlap × Top-K 网格在内存中重建索引，不改正式库；Prompt 对比跑三种系统提示。

详见 [EVALUATION.md](EVALUATION.md)。

## 9. 失败模式

| 场景 | 行为 |
| --- | --- |
| 无 API Key | 自动 mock，UI 显示“离线演示模式” |
| LLM 401/429 | 路由返回 502 或 SSE error，提示检查 Key；不崩溃 |
| Embedding 维度不符 | `EmbeddingError`，提示调整 `EMBEDDING_DIM` 并重建 |
| 扫描版 PDF | 422，提示先 OCR |
| 加密 PDF | 422 |
| 重复 PDF | 按 SHA-256 返回已有论文 |
| pgserver 启动失败 | 默认降级 SQLite（可关闭 `SQLITE_FALLBACK`） |
| 在线检索 DNS/超时 | 收集 errors，仍返回本地结果 |

## 10. 可扩展点

- OCR：接入 PaddleOCR / MinerU，在 `pdf_parser` 文本量不足时触发；
- 重排：接入 bge-reranker-v2-m3；
- 多模态：PDF 图表抽取；
- 鉴权：JWT + 用户级 `owner_id`；
- 任务队列：评测改 Celery/Arq；
- 增量索引：按 `pdf_sha256` + `embedding_model` 判断；
- 引用校验：生成后逐句 NLI 校验。
