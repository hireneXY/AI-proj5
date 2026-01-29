import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from configs.base_config import BaseConfig
from data.dataset import MultimodalDataset

def build_model(vocab_size, emb_dim=128, hidden=128, num_classes=3):
    class TextClassifier(nn.Module):
        def __init__(self, vocab_size, emb_dim, hidden, num_classes):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.fc = nn.Sequential(
                nn.Linear(emb_dim, hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden, num_classes)
            )

        def forward(self, input_ids, attention_mask):
            x = self.embedding(input_ids)  # (B, L, D)
            # mask out padding
            att = attention_mask.unsqueeze(-1).float()
            x = (x * att).sum(dim=1) / (att.sum(dim=1).clamp(min=1.0))
            logits = self.fc(x)
            return logits

    return TextClassifier(vocab_size, emb_dim, hidden, num_classes)

def collate_fn(batch):
    # default batches are fine as dataset returns tensors
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]).long().squeeze(-1),
        "guid": [b["guid"] for b in batch]
    }

def run():
    cfg = BaseConfig()
    cfg.train_file = "data/clean_train.txt"
    cfg.data_dir = "data"
    device = torch.device(cfg.device)

    # load full train list
    train_list = []
    with open(cfg.train_file, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            g, tag = line.split(",", 1)
            train_list.append((g.strip(), tag.strip().lower()))

    random.seed(42)
    random.shuffle(train_list)

    # small subset for smoke-run
    subset_size = min(800, len(train_list))
    subset = train_list[:subset_size]

    ds = MultimodalDataset(subset, cfg, is_train=True)
    # determine vocab size from fallback mapping
    vocab_size = max(getattr(ds, "_next_id", 2), len(getattr(ds, "_word2id", {})) + 1)
    # also check max id in dataset samples
    max_id = 0
    for i in range(len(ds)):
        item = ds[i]
        if isinstance(item["input_ids"], torch.Tensor):
            max_id = max(max_id, int(item["input_ids"].max().item()) if item["input_ids"].numel()>0 else 0)
    vocab_size = max(vocab_size, max_id + 1, 128)

    model = build_model(vocab_size=vocab_size, emb_dim=128, hidden=128, num_classes=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model.train()
    for epoch in range(1):
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * input_ids.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += input_ids.size(0)
        print(f"Epoch {epoch+1} loss={(total_loss/total):.4f} acc={(correct/total):.4f}")

if __name__ == "__main__":
    run()


