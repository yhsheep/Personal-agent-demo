"""示例论文的主题别名，用于提升“原始 RAG / DPR / GraphRAG”等指代检索。"""

SEED_ALIASES: dict[str, list[str]] = {
    "rag": [
        "original RAG",
        "RAG paper",
        "Lewis et al.",
        "retrieval augmented generation",
        "DPR BART",
        "parametric non-parametric memory",
    ],
    "react": ["ReAct agent", "reasoning and acting", "Yao et al.", "tool use agent"],
    "toolformer": ["Toolformer", "API calls", "self-supervised tool learning", "Schick et al."],
    "self-rag": ["Self-RAG", "reflection tokens", "self reflection", "adaptive retrieval"],
    "dpr": ["DPR", "dense passage retrieval", "dual encoder", "in-batch negatives"],
    "colbert": ["ColBERT", "late interaction", "MaxSim", "token-level retrieval"],
    "lost-middle": ["Lost in the Middle", "long context", "position bias", "U-shaped"],
    "cot": ["chain of thought", "CoT", "reasoning prompting", "Wei et al."],
    "graphrag": ["GraphRAG", "graph RAG", "global summarization", "community summaries"],
    "crag": ["CRAG", "corrective RAG", "retrieval evaluator", "web search correction"],
}
