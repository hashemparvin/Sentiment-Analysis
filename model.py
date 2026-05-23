import torch
import torch.nn as nn
from transformers import AutoModel

class GATv2Layer(nn.Module):

    def __init__(self,dim):

        super().__init__()

        self.W = nn.Linear(dim,dim)
        self.attn = nn.Linear(dim*2,1)

    def forward(self,x):

        B,N,D = x.shape

        h = self.W(x)

        h_i = h.unsqueeze(2).expand(B,N,N,D)
        h_j = h.unsqueeze(1).expand(B,N,N,D)

        a = torch.cat([h_i,h_j],dim=-1)

        e = self.attn(a).squeeze(-1)

        alpha = torch.softmax(e,dim=-1)

        out = torch.matmul(alpha,h)

        return out


class GINLayer(nn.Module):

    def __init__(self,dim):

        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim,dim),
            nn.ReLU(),
            nn.Linear(dim,dim)
        )

    def forward(self,x,adj):

        agg = torch.matmul(adj,x)

        out = self.mlp(x + agg)

        return out


class DualGraphModel(nn.Module):

    def __init__(self,
                 num_classes=3,
                 weight_roberta=1.0,
                 weight_sent=1.0,
                 weight_chunk=1.0):

        super().__init__()

        self.encoder = AutoModel.from_pretrained("roberta-base")

        hidden = self.encoder.config.hidden_size

        self.gat1 = GATv2Layer(hidden)
        self.gat2 = GATv2Layer(hidden)

        self.gin1 = GINLayer(hidden)
        self.gin2 = GINLayer(hidden)

        self.weight_roberta = weight_roberta
        self.weight_sent = weight_sent
        self.weight_chunk = weight_chunk

        self.fc = nn.Linear(hidden*3,num_classes)

    def forward(self,ids,mask,sp,cp,adj):

        out = self.encoder(input_ids=ids,attention_mask=mask)

        tok = out.last_hidden_state
        hidden = tok.size(-1)

        cls = tok[:,0]

        sent = torch.gather(tok,1,sp.unsqueeze(-1).expand(-1,-1,hidden))
        sent = self.gat1(sent)
        sent = self.gat2(sent)
        sent = sent.mean(1)

        chunk = torch.gather(tok,1,cp.unsqueeze(-1).expand(-1,-1,hidden))
        chunk = self.gin1(chunk,adj)
        chunk = self.gin2(chunk,adj)
        chunk = chunk.mean(1)

        cls_w = self.weight_roberta * cls
        sent_w = self.weight_sent * sent
        chunk_w = self.weight_chunk * chunk

        fuse = torch.cat([cls_w,sent_w,chunk_w],dim=1)

        logits = self.fc(fuse)

        return logits
