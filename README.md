# Code Search Project (CoSQA)

Simple code search project with three parts:

1. Embedding-based search engine
2. Evaluation on CoSQA (Recall@10, MRR@10, NDCG@10)
3. Fine-tuning on CoSQA + metrics improvement + loss plot

---

## Files

- `search_engine.py`        – encoder + simple in-memory search
- `metrics.py`              – Recall@K, MRR@K, NDCG@K
- `evaluate_cosqa.py`       – eval on CoSQA with base model
- `finetune_cosqa.py`       – fine-tune on CoSQA + re-eval + loss.png
- `requirements.txt`        – Python deps
  
---

