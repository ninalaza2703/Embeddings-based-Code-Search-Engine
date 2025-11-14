from typing import List, Sequence, Dict
import numpy as np


def _rank_of_item(sorted_indices, target_idx):
    positions = np.where(sorted_indices == target_idx)[0]
    if len(positions) == 0:
        return 10 ** 9 
    return int(positions[0]) + 1  


def metrics_for_single_query(
    scores,ground_truth_idx,k = 10):

    # sort descending
    sorted_idx = np.argsort(scores)[::-1]
    rank = _rank_of_item(sorted_idx, ground_truth_idx)

    # Recall@K == hit@K when one relevant per query
    recall_at_k = 1.0 if rank <= k else 0.0

    # MRR@K
    mrr_at_k = 1.0 / rank if rank <= k else 0.0

    # NDCG@K with binary relevance: DCG = 1/log2(rank+1), IDCG = 1
    if rank <= k:
        dcg = 1.0 / np.log2(rank + 1.0)
    else:
        dcg = 0.0
    ndcg_at_k = dcg  # IDCG = 1.0

    return {
        "recall@k": float(recall_at_k),
        "mrr@k": float(mrr_at_k),
        "ndcg@k": float(ndcg_at_k),
    }


def aggregate_metrics(
    all_scores, ground_truth, k= 10):

    assert len(all_scores) == len(ground_truth)
    recalls: List[float] = []
    mrrs: List[float] = []
    ndcgs: List[float] = []

    for scores, gt in zip(all_scores, ground_truth):
        m = metrics_for_single_query(scores, gt, k=k)
        recalls.append(m["recall@k"])
        mrrs.append(m["mrr@k"])
        ndcgs.append(m["ndcg@k"])

    return {
        "Recall@{}".format(k): float(np.mean(recalls)),
        "MRR@{}".format(k): float(np.mean(mrrs)),
        "NDCG@{}".format(k): float(np.mean(ndcgs)),
    }
