import torch
import torch.nn as nn
from torch.nn import functional as F
import wave, struct, glob
import matplotlib.pyplot as plt

#torch.manual_seed(1337)

dirs = list(glob.iglob("kikuwu_img/*/*.png"))

##
## 1. randomize test/train split
##

n = int(0.9*len(dirs)) # first 90% will be train, rest val
train_dirs = dirs[:n]
val_dirs = dirs[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    directory = train_dirs if split == 'train' else val_dirs
    spec = plt.imread(directory[0])
    #......
    #return x, y

class Model(nn.Module):
    pass


class FeedFoward(nn.Module):
    pass

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
            ks = [key(idy.view(B, -1)).view(self.f1, H) for key in self.keys]
            qs = [query(idy.view(B, -1)).view(self.f1, H) for query in self.queries]
            weis = [F.softmax((qs[i] @ ks[i].transpose(-2,-1)).masked_fill(self.tril[:T, :T] == 0, float('-inf')), dim=-1) for i in range(self.f2)]### type: ignore #* (F*T)**-0.5
            vs = [value(idy.view(B, -1)).view(self.f1, H) for value in self.values]
            y = torch.cat([torch.sum(weis[i] @ vs[i], -1, keepdim=True) for i in range(self.f2)], -1)
            ys.append(y)
            #print(x.shape)
            #print(x)
            #return x
            #print(x)
        return torch.stack(ys)

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
            ks = [key(idy.view(B, -1,)).view(self.f1, H) for key in self.keys]
            qs = [query(idy.view(B, -1,)).view(self.f1, H) for query in self.queries]
            weis = [F.softmax((qs[i] @ ks[i].transpose(-2,-1)).masked_fill(self.tril[:T, :T] == 0, float('-inf')), dim=-1) for i in range(self.f2)]### type: ignore #* (F*T)**-0.5
            vs = [value(idy.view(B, -1,)).view(self.f1, H) for value in self.values]
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
        self.heads = nn.ModuleList([Head() for _ in range(4)])
    
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

class Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ffwd = FeedFoward(10)
        self.ln1 = nn.LayerNorm(10)
        self.ln2 = nn.LayerNorm(10)

    def forward(self, x):
        print("g")
        print(x.shape)
        print(self.ln1(x).shape)
        print(self.sa(self.ln1(x)).shape)
        x = x + self.sa(self.ln1(x))
        print(x.shape)
        x = x + self.ffwd(self.ln2(x))
        print(x.shape)
        return x

block = Head()
print(torch.rand(2,2,10).shape)
block(torch.rand(2,2,10))