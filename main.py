import json
import math
import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from tokenizers import Tokenizer
import logging
import argparse
from torch.utils.data import DataLoader

EVAL_PROMPT = "Once upon a time"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="model")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer.json")
    parser.add_argument("--config-path", type=str, default="config.json")
    parser.add_argument("--reset", action="store_true", help="Reset the model")

    args = parser.parse_args()
    
        
    log_file = f"logs/{args.model_name}.log"
    if args.reset:
        os.remove(log_file)
        os.remove(args.model_name)
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    file_logger = logging.getLogger("file_logger")
    file_logger.setLevel(logging.INFO)
    file_logger.propagate = False
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    file_logger.addHandler(file_handler)
    
    with open(args.config_path, "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    datasets_path = {"train": "data/train.txt", "test": "data/test.txt", "val": "data/val.txt"}

        
    BATCH_SIZE = config["batch_size"]        
    SEQ_LEN = config["block_size"]           # block size
    EMBED_DIM = config["embed_dim"]         
    NUM_HEADS = config["num_heads"]       
    NUM_LAYERS = config["num_layers"]       
    MLP_RATIO = config["mlp_ratio"]     
    DROPOUT = config["dropout"]      
    VOCAB_SIZE = config["vocab_size"]     # size of vocab
    LR = config["lr"]          
    EPOCHS = config["epochs"]

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
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        
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
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50):
        self.eval()
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

class StreamingDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, tokenizer, block_size):
        self.path = path
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as f:
            buffer = []
            for line in f:
                tokens = self.tokenizer.encode(line.strip()).ids
                buffer.extend(tokens)
                while len(buffer) >= self.block_size + 1:
                    x = torch.tensor(buffer[:self.block_size], dtype=torch.long)
                    y = torch.tensor(buffer[1:self.block_size+1], dtype=torch.long)
                    yield x, y
                    buffer = buffer[1:]  # glissement de la fenêtre


            
model = myTransformer(EMBED_DIM, NUM_HEADS, SEQ_LEN, VOCAB_SIZE, NUM_LAYERS, MLP_RATIO, DROPOUT)
model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss() 

train_loader = DataLoader(
    StreamingDataset(datasets_path["train"], tokenizer, SEQ_LEN),
    batch_size=BATCH_SIZE
)
val_loader = DataLoader(
    StreamingDataset(datasets_path["val"], tokenizer, SEQ_LEN),
    batch_size=BATCH_SIZE
)
test_loader = DataLoader(
    StreamingDataset(datasets_path["test"], tokenizer, SEQ_LEN),
    batch_size=BATCH_SIZE
)


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, save_every=2):
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for step, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (step + 1) % 5000 == 0:
                avg_loss = running_loss / 100
                tqdm.write(f"[Epoch {epoch}/{epochs}] Step {step+1} | Train loss: {avg_loss:.4f}")
                file_logger.info(f"[Epoch {epoch}/{epochs}] Step {step+1} | Train loss: {avg_loss:.4f}")
                running_loss = 0.0
                
            if (step + 1) % 20000 == 0 or step == 0:
                generated = evaluate_prompt(model, tokenizer, EVAL_PROMPT, device)
                os.makedirs(f"{args.model_name}", exist_ok=True)
                with open(f"{args.model_name}/eval_prompts.txt", "a", encoding="utf-8") as f:
                    f.write(generated + "\n")
                    tqdm.write(f"✅ Generated: {generated}")
                    file_logger.info(f"✅ Generated: {generated}")

        # Validation périodique
        val_loss = evaluate(model, val_loader, criterion, mode=f"Validation Epoch {epoch}")
        epoch_time = time.time() - start_time
        logging.info(f"Epoch {epoch} done in {epoch_time:.1f}s | Val loss: {val_loss:.4f}")
        
        os.makedirs(f"{args.model_name}/checkpoints", exist_ok=True)

        # Sauvegarde si amélioration
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(f"{args.model_name}/checkpoints", exist_ok=True)
            save_path = f"{args.model_name}/checkpoints/{args.model_name}_best.pt"
            torch.save(model.state_dict(), save_path)
            logging.info(f"✅ New best model saved: {save_path}")

        # Sauvegarde périodique
        if epoch % save_every == 0:
            save_path = f"{args.model_name}/checkpoints/{args.model_name}_epoch{epoch}.pt"
            torch.save(model.state_dict(), save_path)
            logging.info(f"💾 Checkpoint saved: {save_path}")
                
@torch.no_grad()
def evaluate(model, dataloader, criterion, mode="Validation"):
    model.eval()
    total_loss = 0.0
    count = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
        count += 1
    avg_loss = total_loss / count
    logging.info(f"[{mode}] Average loss: {avg_loss:.4f}")
    return avg_loss

@torch.no_grad()
def evaluate_prompt(model, tokenizer, eval_prompt, device, max_new_tokens=50):
    model.eval()
    ids = tokenizer.encode(eval_prompt).ids
    input_ids = torch.tensor([ids], dtype=torch.long).to(device)
    out = model.generate(input_ids, max_new_tokens=max_new_tokens)
    decoded = tokenizer.decode(out[0].tolist())
    return decoded



if __name__ == "__main__":
    logging.info("Starting training...")
    os.makedirs(f"{args.model_name}", exist_ok=True)
    with open(f"{args.model_name}/config.json", "w") as f:
        json.dump(config, f)
    train_model(model, train_loader, val_loader, optimizer, criterion, EPOCHS)
    logging.info("Evaluating on test set...")
    evaluate(model, test_loader, criterion, "Test")
    os.makedirs(f"{args.model_name}/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"{args.model_name}/checkpoints/{args.model_name}.pt")
    logging.info("Training complete.")
