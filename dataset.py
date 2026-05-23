import torch
from torch.utils.data import Dataset
from graph import build_graph

class TweetDataset(Dataset):

    def __init__(self,texts,labels,tokenizer):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self,idx):

        text = self.texts[idx]
        label = self.labels[idx]

        ids,mask,sp,cp,adj = build_graph(text,self.tokenizer)

        return ids,mask,sp,cp,adj,label


def collate(batch):

    ids = torch.stack([b[0] for b in batch])
    mask = torch.stack([b[1] for b in batch])
    sp = torch.stack([b[2] for b in batch])
    cp = torch.stack([b[3] for b in batch])
    adj = torch.stack([b[4] for b in batch])
    y = torch.tensor([b[5] for b in batch])

    return ids,mask,sp,cp,adj,y
