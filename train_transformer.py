import torch
import torch.nn as nn
from torch.nn import functional as F
import wave, struct, glob
import matplotlib.pyplot as plt
import random
import numpy as np

#torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 16
length = 24
depth = 3
freq = 2049
segments = 32
encode_size = 8
n_heads = 4
n_blocks = 4

dirs = list(glob.iglob("kikuwu_specs/*.png"))
#random.shuffle(dirs)

##
## 1. randomize test/train split
##

n = int(0.9*len(dirs)) # first 90% will be train, rest val
train_dirs = dirs[:n]
val_dirs = dirs[n:]

# data loading
def get_batch(split):

    directory = train_dirs if split == 'train' else val_dirs
    spec = torch.from_numpy(plt.imread(directory[0])[:,:,0])
    spec = torch.cat([torch.zeros(spec.shape[0], length), spec, torch.zeros(spec.shape[0], length)],dim = -1)
    
    anchor = torch.randint(spec.shape[1] + length * (2**(depth+1) - 1), (batch_size,)) + length
    pos = anchor - length
    y = [(spec[:,anchor[i]:anchor[i]+length] if (anchor[i] >= 0 and anchor[i] < spec.shape[1]) else torch.zeros((spec.shape[0], length))) for i in range(batch_size)]
    x = []
    for b in range(batch_size):
        xa = []
        curr_anchor = anchor[b]
        for i in range(depth):
            curr_anchor -= length * 2**i
            spec_chunk = spec[:,curr_anchor:curr_anchor+length] if (curr_anchor >= 0 and curr_anchor < spec.shape[1]) else torch.zeros((spec.shape[0], length))
            xa.append(spec_chunk)
        xa.reverse()
        x.append(torch.cat(xa, dim=-1))
    x = torch.stack(x)
    y = torch.stack(y)
    '''
    print(x.shape, y.shape)
    plt.imshow(x)
    plt.show()
    plt.imshow(y)
    plt.show()
    plt.imshow(spec)
    plt.show()
    '''
    #......
    x = x.to(device)
    y = y.to(device)
    return x, y, pos

class Head(nn.Module):

    def __init__(self):
        super().__init__()
        self.keys = [Encoder(freq * depth * length, freq * depth * length, bias=False) for _ in range(length)]
        self.queries = [Encoder(freq * depth * length, freq * depth * length, bias=False) for _ in range(length)]
        self.values = [Encoder(freq * depth * length, freq * depth * length, bias=False) for _ in range(length)]
        self.register_buffer('tril', torch.tril(torch.ones(freq, freq)))
        #self.query = nn.Linear(n_embd, head_size, bias=False)
        #self.value = nn.Linear(n_embd, head_size, bias=False)

    def forward(self, x):
        B, H, T = x.shape #Batch, Frequencies, Time
        #print(x.shape)
        #print(x.view(B, -1).shape)
        ks = [key(x.view(B, -1)).view(B, H, T) for key in self.keys]
        qs = [query(x.view(B, -1)).view(B, H, T) for query in self.queries]
        weis = [F.softmax((qs[i] @ ks[i].transpose(-2,-1)).masked_fill(self.tril[:H, :H] == 0, float('-inf')), dim=-1) for i in range(length)]### type: ignore #* (F*T)**-0.5
        vs = [value(x.view(B, -1)).view(B, H, T) for value in self.values]
        y = torch.cat([torch.sum(weis[i] @ vs[i], -1, keepdim=True) for i in range(length)], -1)
        #x = torch.cat([torch.sum(k '''wei''' @ idy '''value''', -1, keepdim=True) for i in range(f2)], -1)
        #print(x.shape)
        #print(x)
        #return x
        #print(x)
        return y

'''
class Head(nn.Module):

    def __init__(self):
        super().__init__()
        self.f1 = 2
        self.f2 = 10
        self.keys = [nn.Linear(self.f1 * 2, self.f1 * 2, bias=False) for _ in range(self.f2)]
        self.queries = [nn.Linear(self.f1 * 2, self.f1 * 2, bias=False) for _ in range(self.f2)]
        self.values = [nn.Linear(self.f1 * 2, self.f1 * 2, bias=False) for _ in range(self.f2)]
        self.register_buffer('tril', torch.tril(torch.ones(self.f1, self.f1)))
        #self.query = nn.Linear(n_embd, head_size, bias=False)
        #self.value = nn.Linear(n_embd, head_size, bias=False)

    def forward(self, x):
        B, H, T = x.shape #Batch, Frequencies, Time
        ys = []
        for idy in x:
            ks = [key(idy.view(-1,)).view(self.f1, H) for key in self.keys]
            qs = [query(idy.view(-1,)).view(self.f1, H) for query in self.queries]
            weis = [F.softmax((qs[i] @ ks[i].transpose(-2,-1)).masked_fill(self.tril[:T, :T] == 0, float('-inf')), dim=-1) for i in range(self.f2)]### type: ignore #* (F*T)**-0.5
            vs = [value(idy.view(-1,)).view(self.f1, H) for value in self.values]
            y = torch.cat([torch.sum(weis[i] @ vs[i], -1, keepdim=True) for i in range(self.f2)], -1)
            ys.append(y)
            #print(x.shape)
            #print(x)
            #return x
            #print(x)
        return torch.stack(ys)'''

class MultiHeadAttention(nn.Module): ###idea: add torch variable multiply for each head
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(n_heads)])
    
    def forward(self, x):
        out = torch.sum(torch.stack([h(x) for h in self.heads]), dim=0)
        #out = self.dropout(self.proj(out))
        return out
    
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)#,
            #nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, in_size, out_size, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, encode_size, bias=bias),
            nn.ReLU(),
            nn.Linear(encode_size, out_size, bias=bias)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ffwd = FeedFoward(length)
        self.ln1 = nn.LayerNorm(depth * length)
        self.ln2 = nn.LayerNorm(depth * length)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.sound_embedding = Encoder(freq * depth * length, freq * depth * length)
        self.absolute_embedding = Encoder(freq * depth * length, freq * depth * length)
        #self.relative embedding = nn.Linear(...)
        self.song_embedding = Encoder(freq * depth * length, freq * depth * length)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_blocks)])
        self.lnorm = nn.LayerNorm(length) # final layer norm
        self.lm_head = nn.Linear(1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, H, T = idx.shape
        print(idx.shape)
        print(self.sound_embedding(idx.view(B, -1)).shape)
        sound_emb = self.sound_embedding(idx.view(B, -1)).view(B, H, T)
        abs_emb = self.absolute_embedding(idx.view(B, -1)).view(B, H, T)
        #rel_emb = self.relative_embedding(idx)
        sound_emb = self.song_embedding(idx.view(B, -1)).view(B, H, T)
        x = sound_emb + abs_emb + sound_emb
        x = self.lnorm(self.blocks(x))
        x = self.lm_head(x.view(B, -1)).view(B, H, T)
        return x

'''
model = Model()
model(torch.rand(2,2,10))
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
'''
model = Model().to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

xb, yb, posb = get_batch('train')
print(xb.shape, yb.shape)
print(model(xb))