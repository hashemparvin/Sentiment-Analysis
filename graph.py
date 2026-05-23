import torch
import nltk
from config import MAX_LEN,NUM_SENT,NUM_CHUNK

def build_graph(text, tokenizer):

    sentences = nltk.sent_tokenize(text)

    enc = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    input_ids = enc["input_ids"].squeeze(0)
    attn = enc["attention_mask"].squeeze(0)

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    sent_ptrs = torch.zeros(NUM_SENT).long()
    chunk_ptrs = torch.zeros(NUM_CHUNK).long()

    sent_idx = 0

    for i,t in enumerate(tokens):
        if t in [".","!","?"] and sent_idx < NUM_SENT-1:
            sent_idx += 1
            sent_ptrs[sent_idx] = min(i+1,MAX_LEN-1)

    step = MAX_LEN // NUM_CHUNK

    for i in range(NUM_CHUNK):
        chunk_ptrs[i] = i*step

    adj = torch.ones(NUM_CHUNK,NUM_CHUNK)
    adj = adj / adj.sum(dim=1,keepdim=True)

    return input_ids, attn, sent_ptrs, chunk_ptrs, adj
