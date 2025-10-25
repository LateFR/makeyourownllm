import math
from tqdm import tqdm
import torch
import torch.nn as nn
from tokenizers import Tokenizer
import logging
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="model")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer.json")
    
    args = parser.parse_args()
    
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

datasets_path = {"train": "data/train.txt", "test": "data/test.txt", "val": "data/val.txt"}
    
if not args.tokenizer_path:
    raise ValueError("Please provide a tokenizer path with --tokenizer-path")
tokenizer = Tokenizer.from_file(args.tokenizer_path)
    
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads # dimension of each head ex: embed_dim = 512 and num_heads = 8 -> head_dim = 64
        
        # linear projection to get each head's query/key/value
        self.qvk_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        
        # linear projection to get each head's output
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # casual mask
        self.register_buffer("causal_mask", torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0)) # out -> (1, 1, block_size, block_size)
        
    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        # output: (batch_size, seq_len, embed_dim)
        
        B, T, C = x.shape
        
        qkv = self.qkv_proj(x)  # -> (B, T, 3*C)    the result is in the 3rd dimension
        
        # we extract the query, key and value from the last dimension
        q, k, v = qkv.chunk(3, dim=-1)
        
        # each head will process head_dim dimensions
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # -> (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # mask out the causal mask
        # the causal mask is a lower triangular matrix with zeros on the diagonal
        # it blocks the attention from looking forward in time
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf")) 
        
        
        att = torch.softmax(att, dim=-1) # normalize the attention scores (total sum of each row is now 1)
        
        out = torch.matmul(att, v) # ponderate the value vectors by the attention scores
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)# -> (B, T, C)
        
        out = self.out_proj(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__ (self, embed_dim, num_heads, block_size, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(embed_dim)
        
        self.attn = CausalSelfAttention(embed_dim, num_heads, block_size)
        
        self.ln2 = nn.LayerNorm(embed_dim)
        
        self.hidden_dim = int(embed_dim * mlp_ratio)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        # output: (batch_size, seq_len, embed_dim)
        
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.mlp(self.ln2(x))
        
        return x
    
    
class myTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, vocab_size, num_layers, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.pos_embedding = nn.Embedding(block_size, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, block_size, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False) # we project back to the vocabulary size
        
    def forward(self, idx):
        # idx: (batch_size, seq_len)
        # output: (batch_size, seq_len, embed_dim)
        
        B, T = idx.shape
        token_emb = self.token_embedding(idx) # shape (B, T, embed_dim) 
        pos = torch.arange(T, device=token_emb.device)
        pos_emb = self.pos_embedding(pos).unsqueeze(0) # shape (1, T, embed_dim)
        x = token_emb + pos_emb # broadcasting -> (B, T, embed_dim)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.lm_head(x) # (B, T, vocab_size)
        
        return x

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, block_size):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        tokens = tokenizer.encode(text).ids
        self.inputs = []
        self.targets = []
        for i in range(0, len(tokens) - block_size):
            self.inputs.append(torch.tensor(tokens[i:i+block_size]))
            self.targets.append(torch.tensor(tokens[i+1:i+block_size+1]))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

            
BATCH_SIZE = 4        # taille d'un batch
SEQ_LEN = 64          # longueur de la séquence (block size)
EMBED_DIM = 64        # dimension de l'embedding
NUM_HEADS = 4         # nombre de têtes d'attention
NUM_LAYERS = 2        # nombre de blocks transformer
MLP_RATIO = 4.0       # ratio pour la dimension du MLP
DROPOUT = 0.1         # dropout
VOCAB_SIZE = 30000     # size of vocab
LR = 3e-4             # learning rate
EPOCHS = 10            # nombre d'époques
model = myTransformer(EMBED_DIM, NUM_HEADS, SEQ_LEN, VOCAB_SIZE, NUM_LAYERS, MLP_RATIO, DROPOUT)
model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss() 

train_dataset = TextDataset("data/train.txt", tokenizer, SEQ_LEN)
def train_model(model, dataloader, optimizer, criterion, EPOCHS):
    for epoch in tqdm(range(EPOCHS)):
        model.train() # mode train
        running_loss = 0.0
        for step, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            
            logits = model(x) # (B, T, vocab_size)
            
            # reshape and loss
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (step + 1) % 10 == 0:
                tqdm.write(f"Epoch [{epoch+1}/{EPOCHS}] Step [{step+1}/{len(dataloader)}] Loss: {running_loss/10:.4f}")
                running_loss = 0.0
                
@torch.no_grad()
def evaluate(model, dataloader, criterion, mode="Validation"):
    model.eval()
    total_loss = 0
    for x, y in tqdm(dataloader, desc=mode):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
    return total_loss / len(dataloader)