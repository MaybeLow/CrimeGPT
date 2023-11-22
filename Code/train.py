import torch.nn as nn
import torch
import numpy as np
from gpt_model import GPT
from config import get_batch_dims 
from tokenization import get_subword_encoding
from nltk.tokenize import word_tokenize
import os

batch_num, batch_size = get_batch_dims()
eval_it = 40
learning_rate = 3e-4
max_iters = 2000
eval_interval = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

enc = get_subword_encoding()

os.chdir("../Data")
with open('book.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(set(chars))
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] + ' ' for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
vocab_size = len(set(chars))

n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    if(split == "train"):
        split_data = train_data
    else:
        split_data = val_data

    random_indexes = torch.randint(len(split_data)-batch_size, (batch_num, ))
    batches = torch.stack([split_data[i:i+batch_size] for i in random_indexes])
    targets = torch.stack([split_data[i+1:i+1+batch_size] for i in random_indexes])

    batches, targets = batches.to(device), targets.to(device)
    return batches, targets

a, b = get_batch("train")

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_it)
        for k in range(eval_it):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = GPT(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
torch.save(m, 'model.pth')
generated_text = m.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated_text))