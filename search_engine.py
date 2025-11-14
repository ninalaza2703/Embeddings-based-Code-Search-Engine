from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


BASE_MODEL_NAME = "microsoft/codebert-base"  


class HFEncoder:

    def __init__(
        self,
        model_name: str = BASE_MODEL_NAME,
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if checkpoint_path is not None:
            print(f"[HFEncoder] Loading fine-tuned weights from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 16,
        max_length: int = 256,
        normalize: bool = True,
    ) -> np.ndarray:
        all_embs: List[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = self.tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            outputs = self.model(**encoded)
            # CLS embedding
            cls_emb = outputs.last_hidden_state[:, 0]  # (batch, hidden)
            if normalize:
                cls_emb = torch.nn.functional.normalize(cls_emb, p=2, dim=-1)
            all_embs.append(cls_emb.cpu().numpy())

        return np.vstack(all_embs)


@dataclass
class SearchResult:
    score: float
    doc: str
    index: int


class EmbeddingSearchEngine:

    def __init__(self, encoder: HFEncoder, documents: Sequence[str]) -> None:
        self.encoder = encoder
        self.documents: List[str] = list(documents)
        print(f"[SearchEngine] Encoding {len(self.documents)} documents...")
        self.doc_embeddings: np.ndarray = self.encoder.encode(self.documents)  # (N, d)
        # Embeddings are already normalized; dot product = cosine similarity.

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        query_emb = self.encoder.encode([query])[0]  # (d,)
        scores = self.doc_embeddings @ query_emb  # (N,)

        top_k = min(top_k, len(self.documents))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[SearchResult] = []
        for idx in top_indices:
            results.append(
                SearchResult(score=float(scores[idx]), doc=self.documents[idx], index=int(idx))
            )
        return results

    docs = [
        "def add(a, b): return a + b",
        "def multiply(a, b): return a * b",
        "def factorial(n): return 1 if n == 0 else n * factorial(n-1)",
        "def sort_list(lst): return sorted(lst)",
    ]

    encoder = HFEncoder(model_name=BASE_MODEL_NAME)
    engine = EmbeddingSearchEngine(encoder, docs)

    query = "function that calculates product"
    print(f"\nQuery: {query}")
    results = engine.search(query, top_k=3)

    for r in results:
        print(f"Score: {r.score:.4f} | idx={r.index} | doc={r.doc}")

