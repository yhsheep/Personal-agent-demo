# data/

| 目录 / 文件 | 是否提交 | 说明 |
| --- | --- | --- |
| `seed/papers.json` | 是 | 10 篇示例论文元数据、摘要与离线笔记 |
| `seed/pdfs/` | 可选 | `seed --download` 下载的真实 arXiv PDF 缓存（体积较大，可自行忽略） |
| `eval/dataset.jsonl` | 是 | 25 条中文评测集 |
| `uploads/` | 否 | 上传/导入的 PDF，运行后生成 |
| `pgdata/` | 否 | 嵌入式 PostgreSQL 数据目录 |
| `logs/` | 否 | 运行日志 |
| `wenlan.db` | 否 | 仅 SQLite 测试降级时生成 |

首次运行：

```bash
python -m app.cli init-db
python -m app.cli seed
```

`seed` 默认生成可离线使用的摘要笔记 PDF；`seed --download` 会尝试下载真实 arXiv PDF，失败自动回退。
