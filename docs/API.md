# API 示例

Base URL：`http://127.0.0.1:8000`  
交互式文档：`/api/docs`

## 健康与配置

```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/system/info
```

## 上传 PDF

```bash
curl -X POST http://127.0.0.1:8000/api/papers/upload \
  -F "file=@paper.pdf" \
  -F "title=Optional Title" \
  -F "authors=Alice, Bob"
```

响应包含 `paper.id`、`chunk_count`、`duplicate`。

## 文献列表 / 详情 / Chunk

```bash
curl "http://127.0.0.1:8000/api/papers?limit=20&q=RAG"
curl "http://127.0.0.1:8000/api/papers/<PAPER_ID>"
curl "http://127.0.0.1:8000/api/papers/<PAPER_ID>/chunks?limit=20"
```

PDF：`GET /api/papers/<PAPER_ID>/pdf`。

## 重建索引

```bash
curl -X POST http://127.0.0.1:8000/api/papers/<PAPER_ID>/reindex \
  -H "Content-Type: application/json" \
  -d '{"chunk_size":700,"overlap":50}'
```

全部重建：`POST /api/library/reindex`。

## 导入示例库

```bash
curl -X POST "http://127.0.0.1:8000/api/library/seed?download=false"
```

`download=true` 会尝试下载真实 arXiv PDF，失败自动使用离线笔记。

## RAG 检索

```bash
curl -X POST http://127.0.0.1:8000/api/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query":"RAG 的生成器用了什么模型？","top_k":6}'
```

响应：

```json
{
  "count": 6,
  "results": [
    {
      "chunk_id": "...",
      "paper_id": "...",
      "paper_title": "Retrieval-Augmented Generation for ...",
      "section": "2 Method",
      "page_start": 1,
      "page_end": 1,
      "content": "The generator is BART-large ...",
      "score": 0.41,
      "vector_score": 0.28,
      "lexical_score": 0.52
    }
  ],
  "citations": [...]
}
```

## 找论文

```bash
curl "http://127.0.0.1:8000/api/discovery/search?q=retrieval%20augmented%20generation&limit=8&scope=all"
```

`scope`：`all` / `local` / `online`。

导入：

```bash
curl -X POST http://127.0.0.1:8000/api/discovery/import \
  -H "Content-Type: application/json" \
  -d '{
    "title": "...",
    "authors": ["..."],
    "abstract": "...",
    "year": 2024,
    "venue": "arXiv",
    "arxiv_id": "2401.00001",
    "url": "https://arxiv.org/abs/2401.00001",
    "source": "arxiv",
    "fetch_pdf": true
  }'
```

## 会话与问答

新建会话：

```bash
curl -X POST http://127.0.0.1:8000/api/conversations \
  -H "Content-Type: application/json" -d '{"title":"RAG 精读"}'
```

非流式：

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"RAG 的生成器用了什么模型？","conversation_id":null,"top_k":6}'
```

流式（SSE）：

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"Self-RAG 与原始 RAG 的区别？","conversation_id":null,"top_k":6,"prompt_variant":"citation_strict"}'
```

会话历史：

```bash
curl http://127.0.0.1:8000/api/conversations
curl http://127.0.0.1:8000/api/conversations/<CONVERSATION_ID>
```

## 评测

```bash
curl -X POST http://127.0.0.1:8000/api/evaluations/run \
  -H "Content-Type: application/json" \
  -d '{"kind":"full","limit":25,"top_k":6,"name":"baseline"}'
```

`kind`：

- `full`：检索 + Agent；
- `retrieval`：只跑检索；
- `agent`：只跑 Agent；
- `retrieval_grid`：Chunk/Overlap/Top-K 网格；
- `prompt_grid`：Prompt 对比。

轮询：

```bash
curl http://127.0.0.1:8000/api/evaluations/runs/<RUN_ID>
curl http://127.0.0.1:8000/api/evaluations/runs
```

评测集：

```bash
curl http://127.0.0.1:8000/api/evaluations/dataset?limit=100
```

## 错误格式

```json
{"detail": "PDF 中未生成有效文本块"}
```

常见状态码：

| 状态 | 含义 |
| ---: | --- |
| 400 | 参数/文件类型错误 |
| 404 | 论文/会话/评测不存在 |
| 413 | 上传超过大小限制 |
| 422 | PDF 解析失败、加密、扫描版 |
| 502 | LLM 上游调用失败 |
