import torch
import torch.nn as nn
from torch.nn import functional as F
import wave, struct, glob
import matplotlib.pyplot as plt

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 512 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
eval_iters = 200
n_embd = block_size #64
n_head = 8
n_layer = 8
dropout = 0.2
encoder_depth = 3
encoder_scale = 8
INTMAX = 2**31 - 1
INTMIN = -2**31
# ------------

torch.manual_seed(1337)

dirs = list(glob.iglob("kikuwu_img/*/*.png"))

##
## 1. randomize test/train split
##

waveform = open("waveform.txt", 'r').readline().split(' ')[:-1]
data = torch.tensor([int(item) for item in waveform]) / INTMAX

n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size* encoder_scale, (batch_size,))
    x = torch.stack([data[i:i+block_size* encoder_scale] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            #print(X.shape, "X shape")
            #print(Y.shape, "Y shape")
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Encoder(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()
        self.layers = nn.Sequential(*[nn.Linear(int(in_size * (out_size / in_size)**(i/encoder_depth)), int(in_size * (out_size / in_size)**((i + 1)/encoder_depth))) for i in range(encoder_depth)])
    
    def forward(self, x):
        out = self.layers(x)
        return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # type: ignore # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #print(x.shape, "x.shape")
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.encoder = Encoder(block_size* encoder_scale, block_size)
        self.waveform_embedding_table = nn.Linear(block_size, block_size)
        self.position_embedding_table = nn.Linear(block_size, block_size)#nn.Embedding(block_size, block_size)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, n_embd)

    def forward(self, idx, targets=None):
        #idx = self.encoder(idx)
        #print(idx.shape)
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers

        #tok_emb = self.token_embedding_table(idx) # (B,T,C)
        #print(idx.shape, "idx.shape")
        #print(torch.arange(T, device=device).repeat(batch_size).reshape(batch_size, block_size).shape, "pos.shape")
        tok_emb = self.waveform_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device, dtype=torch.float32))#.repeat(batch_size).reshape(batch_size, block_size)) # (T,C)

        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        output = self.lm_head(x) # (B,T,vocab_size)
        #print(output.shape, "output.shape")

        if targets is None:
            loss = None
        else:
            T, C = output.shape
            output = output.view(T*C)
            targets = targets.view(T*C)
            loss = F.mse_loss(output, targets)

        return output, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for count in range(max_new_tokens):

            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size* encoder_scale:]
            # get the predictions
            #####print(idx_cond, "idx_cond")
            logits, loss = self(idx_cond)

            # focus only on the last time step
            logits = logits[:,-1:] # becomes (B, C)
            '''
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            '''
            #print(idx.shape)
            #print(logits.shape)
            idx = torch.cat((idx, logits), dim=1) # (B, T+1) #idx_next), dim=1) # (B, T+1)

            if count % 1000 == 0:
                print("Generating... {}/{}".format(count, max_new_tokens))

        return idx

model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

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

    #print(logits[0], logits.shape, "logits")
    #print(xb[0], xb.shape, "xb")
    #print(yb[0], yb.shape, "yb")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()    


# generate from the model
context = torch.zeros((1, block_size* encoder_scale), dtype=torch.float32, device=device)
data = torch.clamp(m.generate(context, max_new_tokens=30000)[0], min=-1.0, max=1.0).tolist()
#print(data)
#print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))

with wave.open('sample.wav', 'wb') as sample:
    sample.setnchannels(1)
    sample.setsampwidth(4)
    sample.setframerate(4800)
    for item in data:
        sample.writeframesraw(struct.pack('<l', int(item * INTMAX)))

    sample.close()

plt.plot(data)
plt.show()