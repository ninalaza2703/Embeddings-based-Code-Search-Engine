import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "microsoft/codebert-base"


class HFEncoder:
    def __init__(self, model_name=MODEL_NAME, checkpoint_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts, batch=16):
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tokenizer(
                texts[i:i+batch],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)
            hidden = self.model(**enc).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=-1)
            out.append(hidden.cpu().numpy())
        return np.vstack(out)


class EmbeddingSearchEngine:
    def __init__(self, encoder, docs):
        self.encoder = encoder
        self.docs = list(docs)
        self.doc_emb = encoder.encode(self.docs)

    def search(self, query, top_k=10):
        q = self.encoder.encode([query])[0]
        scores = self.doc_emb @ q
        idxs = np.argsort(scores)[::-1][:top_k]
        return [(float(scores[i]), self.docs[i], i) for i in idxs]


if __name__ == "__main__":
    docs = [
        "def add(a,b): return a+b",
        "def multiply(a,b): return a*b",
        "def factorial(n): return 1 if n==0 else n*factorial(n-1)",
        "def sort_list(x): return sorted(x)"
    ]

    enc = HFEncoder()
    se = EmbeddingSearchEngine(enc, docs)

    r = se.search("function that multiplies numbers")
    for x in r:
        print(x)
