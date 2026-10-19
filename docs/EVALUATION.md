# 评测方法与实验报告

## 1. 目标

评测回答三个问题：

1. **检索找得准吗？** —— Recall@K、MRR、nDCG@K；
2. **Agent 选对工具了吗？** —— Tool Calling Accuracy、Exact Match、Tool F1；
3. **最终任务完成了吗？** —— Task Success Rate、Citation Hit Rate。

所有指标可由 `POST /api/evaluations/run` 重跑，结果持久化到 `evaluation_runs`，前端展示逐用例诊断。

## 2. 评测集

`data/eval/dataset.jsonl`，25 条中文用例，覆盖 10 篇论文：

| 类型 | 数量 | 示例 |
| --- | ---: | --- |
| `rag_qa` 原文问答 | 21 | RAG 的生成器用了什么模型？ |
| `paper_search` 找论文 | 2 | 帮我找几篇讨论 RAG 的论文 |
| `metadata` 元数据 | 1 | Lost in the Middle 的作者是谁？ |
| `multi_turn` 多轮指代 | 1 | 其中 ISSUP 表示什么？ |

其中 `rag_qa` 包含对比、局限、多跳等难度更高的问法。

用例结构：

```json
{
  "id": "q01",
  "type": "rag_qa",
  "difficulty": "easy",
  "question": "RAG 原始论文中，生成器使用的是什么模型？",
  "expected_tools": ["search_library"],
  "relevant_papers": ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"],
  "keyword_groups": [["BART"], ["生成器", "generator"]],
  "reference_answer": "RAG 使用 DPR 检索器与 BART 生成器..."
}
```

- `expected_tools`：必需工具集合；
- `relevant_papers`：检索 ground truth（论文级）；
- `keyword_groups`：每组至少命中一个词，才视为事实通过；
- `reference_answer`：人工参考答案，用于失败诊断与可选 LLM judge。

## 3. 指标定义

### 3.1 Recall@K

对问题 $q$，相关论文集合 $R_q$，Top-K 检索命中的不同论文集合 $H_q$：

$$
\text{Recall@K} = \frac{|R_q \cap H_q|}{|R_q|}
$$

一个相关论文只要任意一个 chunk 进入 Top-K 即命中。多论文问题的 Recall 为比例。

### 3.2 MRR

首个相关 chunk 的排名 $r_q$：

$$
\text{MRR} = \frac{1}{N}\sum_q \frac{1}{r_q}
$$

### 3.3 nDCG@K

按论文去重后的排序计算二值相关性：

$$
\text{DCG@K} = \sum_{i=1}^{K}\frac{rel_i}{\log_2(i+1)},\quad
\text{nDCG@K}=\frac{DCG@K}{IDCG@K}
$$

### 3.4 Tool Calling Accuracy

主指标采用 **必需工具覆盖率**：

$$
\text{ToolAcc} = \frac{1}{N}\sum_q \mathbb{1}[E_q \subseteq A_q]
$$

其中 $E_q$ 是期望工具集合，$A_q$ 是实际调用过的工具集合。  
同时报告：

- `tool_exact_match`：集合完全相等；
- `tool_f1`：按实际/期望工具集合计算 P/R/F1。

这样既不会因为模型多检索一次就判错，也能惩罚漏掉关键工具。

### 3.5 Task Success Rate

逐用例满足以下全部条件记为成功：

1. `keyword_groups` 每组至少命中一个词；
2. 需要 RAG 的用例，至少一条被模型实际引用（`cited=true`）的来源属于 `relevant_papers`；
3. 没有在需要证据时回答“没有足够证据”；
4. 工具/LLM 执行没有报错。

这是一个**确定性、可复现**的规则评测。接入真实 LLM 后，可在 `EvaluationService` 中扩展 LLM-as-judge 作为补充，不替代规则指标。

### 3.6 Citation Hit Rate

$$
\text{CitationHit} = \frac{1}{N}\sum_q \mathbb{1}[\text{被引用的论文} \cap R_q \neq \varnothing]
$$

只对需要引用的用例计算（元数据/找论文不强制引用）。

## 4. 基线结果

运行时间：2026-10-20  
运行配置：`kind=full, limit=25, top_k=6, prompt=citation_strict`  
环境：10 篇示例论文 / 45 个 chunk，离线 Mock Agent + 本地哈希 Embedding。

| 指标 | 数值 |
| --- | ---: |
| Recall@6 | 1.0000 |
| MRR | 0.9800 |
| nDCG@6 | 0.9645 |
| Precision@6 | 0.3813 |
| Hit Rate | 1.0000 |
| Tool Calling Accuracy | 1.0000 |
| Tool Exact Match | 1.0000 |
| Tool F1 | 1.0000 |
| Task Success Rate | 0.9600 |
| Citation Hit Rate | 0.8800 |
| 检索平均延迟 | 2.04 ms |
| Agent 平均延迟 | 3.24 ms |

> 注意：这些数字用于验证**评测管线与系统行为**，不是对某个大模型能力的宣称。MockLLM 是可预测规则策略，真实模型在 Tool Calling 与长答案组织上通常更强，也可能更不稳定。

### 逐用例失败

唯一失败：`q19 ReAct 与 Chain-of-Thought 的关键区别是什么？`  
原因：Mock 抽取式回答缺失关键词“外部/工具/行动”，但引用了正确论文，检索 Recall 与工具选择均正确。  
说明：这是 mock 答案组织的局限，而非检索失败。

## 5. Chunk × Top-K 网格实验

固定语料与 25 条用例，在内存中按 3×2×3 组合重新分块、向量化、检索。  
不修改正式索引，避免实验污染知识库。

| Chunk | Overlap | Top-K | Recall@K | MRR | nDCG@K | Chunks |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | 50 | 3/5/8 | 0.9667 | 0.9400 | 0.9362 | 89 |
| 300 | 100 | 8 | 0.9800 | 0.9333 | 0.9305 | 100 |
| 500 | 50 | 8 | 0.9800 | 0.9000 | 0.9050 | 54 |
| 500 | 100 | 8 | 0.9800 | 0.9133 | 0.9163 | 59 |
| **700** | **50** | **8** | **1.0000** | 0.9333 | **0.9405** | 42 |
| 700 | 100 | 8 | 1.0000 | 0.9333 | 0.9405 | 42 |

### 解读

- **Top-K=8 始终优于 3/5**：在 45 chunk 的小语料上，扩大上下文成本低、收益稳定；
- **Chunk=700** 召回最高，因为每篇论文的摘要笔记较长，600/700 能保留完整事实句；
- **Chunk=300** 的 MRR 更高（0.94）：小块定位更精确，首个相关结果更靠前，但 chunk 数量约翻倍；
- Overlap 从 50 增加到 100 对结果影响很小，却增加约 10% 存储，**50 更划算**；
- 当前语料推荐：**Chunk 700 / Overlap 50 / Top-K 8**（召回优先）；若线上更关注首条命中与上下文成本，选择 **300 / 50 / 3**。

## 6. Prompt 对比

| Prompt | Task Success | Tool Accuracy | Citation Hit |
| --- | ---: | ---: | ---: |
| 严格引用 | 1.0000 | 1.0000 | 1.0000 |
| 深度分析 | 1.0000 | 1.0000 | 1.0000 |
| 简洁回答 | 1.0000 | 1.0000 | 1.0000 |

离线 Mock 模型不消费 system prompt 的生成约束，因此三者相同。**这是预期行为，不是评测缺陷**。  
接入 DeepSeek / 通义千问后，Prompt 会真实影响答案长度、引用密度与分析深度，此时再运行 `kind=prompt_grid` 才有区分度。

## 7. 如何复现

### Web UI

1. 打开“评测实验”；
2. 选择“完整评测”，用例上限 25，Top-K 6；
3. 点击“开始实验”，等待状态从 running 变为 completed；
4. 选择“Chunk × Top-K 网格实验”，保持默认参数运行。

### CLI

```bash
python -m app.cli eval --kind full --limit 25 --top-k 6
python -m app.cli eval --kind retrieval --limit 25 --top-k 6
python -m app.cli experiments --kind retrieval_grid --limit 25
python -m app.cli experiments --kind prompt_grid --limit 10
```

### API

```bash
curl -X POST http://127.0.0.1:8000/api/evaluations/run \
  -H "Content-Type: application/json" \
  -d '{"kind":"full","limit":25,"top_k":6,"name":"baseline"}'
```

轮询：

```bash
curl http://127.0.0.1:8000/api/evaluations/runs/<RUN_ID>
```

## 8. 指标之外：还应关注什么

- **引用准确率**：引用存在不等于引用支持结论，后续可加 NLI/LLM 逐句校验；
- **拒答质量**：没有证据时是否明确说不知道，而不是给“看起来合理”的答案；
- **多轮一致性**：追问“它的方法呢”是否保持指代对象；
- **工具成本**：平均工具调用次数、检索延迟、token 成本；
- **真实语料鲁棒性**：双栏 PDF、公式、表格、扫描件会显著影响解析与分块，需要 OCR/版面分析。

## 9. 已知偏差

1. 示例语料是**真实论文元数据 + 离线摘要笔记 PDF**，不是完整论文全文。真实 PDF 的 chunk 更长、噪声更多，指标会下降；
2. 本地哈希 Embedding 只是可解释的离线基线，语义能力远弱于 bge-m3；
3. 关键词规则可能奖励“堆词”，因此同时要求引用落地，并保留人工参考答案用于抽查；
4. 25 条用例规模较小，适合工程回归，不足以做统计显著性检验。
