import torch
from torch.nn.functional import softmax
import numpy as np

def train_one_epoch(
    model,
    data_loader,
    optimizer,
    criterion,
    device,
    scaler=None,
    use_amp=False
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in data_loader:

        input_ids, attention_mask, sent_ptrs, chunk_ptrs, adj, labels = [
            x.to(device) for x in batch
        ]

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask, sent_ptrs, chunk_ptrs, adj)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids, attention_mask, sent_ptrs, chunk_ptrs, adj)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc

@torch.no_grad()
def eval_one_epoch(
    model,
    data_loader,
    criterion,
    device,
    return_probs=False
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []
    all_probs = []

    for batch in data_loader:

        input_ids, attention_mask, sent_ptrs, chunk_ptrs, adj, labels = [
            x.to(device) for x in batch
        ]

        logits = model(input_ids, attention_mask, sent_ptrs, chunk_ptrs, adj)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

        if return_probs:
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().tolist())

    avg_loss = total_loss / total
    acc = correct / total

    if return_probs:
        return avg_loss, acc, all_labels, all_preds, all_probs

    return avg_loss, acc, all_labels, all_preds



class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, path="best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path

        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss, model):

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0

            # ذخیره بهترین مدل
            torch.save(model.state_dict(), self.path)

        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True
