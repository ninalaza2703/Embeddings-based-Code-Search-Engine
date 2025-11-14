import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import matplotlib.pyplot as plt

from search_engine import HFEncoder, MODEL_NAME
from evaluate_cosqa import evaluate

FT_PATH = "cosqa_finetuned.pt"

class PairDataset(Dataset):
    def __init__(self, q, c):
        self.q = q
        self.c = c
    def __len__(self): return len(self.q)
    def __getitem__(self, i): return self.q[i], self.c[i]

class BiEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(MODEL_NAME)

    def encode(self, ids, mask):
        out = self.model(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return nn.functional.normalize(out, p=2, dim=-1)

    def forward(self, q, c, temp=0.05):
        q_emb = self.encode(q["input_ids"], q["attention_mask"])
        c_emb = self.encode(c["input_ids"], c["attention_mask"])
        logits = q_emb @ c_emb.T / temp
        labels = torch.arange(len(q_emb), device=logits.device)
        return nn.CrossEntropyLoss()(logits, labels)

def get_loader(max_samples=20000, batch=16):
    ds = load_dataset("cosqa", split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    q, c = ds["nl"], ds["code"]
    d = PairDataset(q, c)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def collate(x):
        qs, cs = zip(*x)
        q_enc = tok(list(qs), padding=True, truncation=True, max_length=256, return_tensors="pt")
        c_enc = tok(list(cs), padding=True, truncation=True, max_length=256, return_tensors="pt")
        return q_enc, c_enc

    return DataLoader(d, batch_size=batch, shuffle=True, collate_fn=collate), tok

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader, _ = get_loader()

    model = BiEncoder().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)

    steps = 0
    loss_hist = []
    model.train()

    for q_enc, c_enc in loader:
        q_enc = {k: v.to(device) for k, v in q_enc.items()}
        c_enc = {k: v.to(device) for k, v in c_enc.items()}

        opt.zero_grad()
        loss = model(q_enc, c_enc)
        loss.backward()
        opt.step()

        steps += 1
        if steps % 50 == 0:
            loss_hist.append((steps, float(loss)))
            print(steps, float(loss))
        if steps == 1000:
            break

    torch.save(model.model.state_dict(), FT_PATH)

    xs, ys = zip(*loss_hist)
    plt.plot(xs, ys)
    plt.title("Training Loss")
    plt.savefig("loss.png")

def compare():
    print("Before fine-tuning:")
    base = HFEncoder()
    evaluate(base)

    print("\nTraining...")
    train()

    print("\nAfter fine-tuning:")
    ft = HFEncoder(checkpoint_path=FT_PATH)
    evaluate(ft)

if __name__ == "__main__":
    compare()
