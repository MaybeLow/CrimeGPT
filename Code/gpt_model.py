import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np
import math
from config import get_batch_dims 

embed_dim = 400
num_heads = 4
num_blocks = 4
batch_num, batch_size = get_batch_dims()
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Head(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.head_size = head_size
        self.q = nn.Linear(embed_dim, head_size, bias=False)
        self.k = nn.Linear(embed_dim, head_size, bias=False)
        self.v = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(batch_size, batch_size)))

    def forward(self, x):
        B, T, C = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        x = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        x = x.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        x = F.softmax(x, dim=-1)

        out = x @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(embed_dim // num_heads) for _ in range(num_heads)])
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.concat([head(x) for head in self.heads], dim=-1)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention()
        self.fc = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.fc(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        # Text / Position Embedding
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(batch_size, embed_dim)
        self.blocks = nn.Sequential(*[Block() for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        _, T = idx.shape
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.linear(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -batch_size:]
            # print(idx)
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
