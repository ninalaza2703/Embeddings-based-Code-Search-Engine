import numpy as np
from datasets import load_dataset
from search_engine import HFEncoder, MODEL_NAME
from metrics import aggregate

DATASET_NAME = "gonglinyuan/CoSQA"
QUERY_COL = "doc"
CODE_COL = "code"


def load_split(split="test", max_samples=1000):
    ds = load_dataset(DATASET_NAME, split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    q = ds[QUERY_COL]
    c = ds[CODE_COL]
    gt = list(range(len(q)))
    return q, c, gt

def evaluate(encoder, split="test", max_samples=1000, k=10):
    q, c, gt = load_split(split, max_samples)
    q_emb = encoder.encode(q)
    c_emb = encoder.encode(c)

    all_scores = []
    for i in range(len(q)):
        all_scores.append(q_emb[i] @ c_emb.T)

    m = aggregate(all_scores, gt, k)
    for x, y in m.items():
        print(f"{x}: {y:.4f}")
    return m

if __name__ == "__main__":
    enc = HFEncoder(MODEL_NAME)
    evaluate(enc)
