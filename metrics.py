import numpy as np

def rank_of(sorted_ids, target):
    p = np.where(sorted_ids == target)[0]
    if len(p) == 0:
        return 10**9
    return int(p[0]) + 1

def metrics_single(scores, gt, k=10):
    s = np.argsort(scores)[::-1]
    r = rank_of(s, gt)

    recall = 1.0 if r <= k else 0.0
    mrr = 1.0 / r if r <= k else 0.0
    ndcg = (1.0 / np.log2(r + 1)) if r <= k else 0.0
    return recall, mrr, ndcg

def aggregate(all_scores, ground_truth, k=10):
    rec, mrr, ndcg = [], [], []
    for s, gt in zip(all_scores, ground_truth):
        a, b, c = metrics_single(s, gt, k)
        rec.append(a)
        mrr.append(b)
        ndcg.append(c)
    return {
        f"Recall@{k}": float(np.mean(rec)),
        f"MRR@{k}": float(np.mean(mrr)),
        f"NDCG@{k}": float(np.mean(ndcg))
    }
