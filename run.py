import json
import math
import os
import shutil
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from tokenizers import Tokenizer
import logging
import argparse
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="model")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer.json")
    parser.add_argument("--config-path", type=str, default="config.json")
    parser.add_argument("--reset", action="store_true", help="Reset the model current model and logs. To use if you had run the training script before and want to restart from scratch")
    parser.add_argument("--generate-tokenizer", action="store_true", help="Generate the tokenizer")
    parser.add_argument("--import-datasets", action="store_true", help="This will import the datasets from the datasets.json file, (downloading them from hungging face if necessary). Automatically called if --generate-tokenizer or --merge-datasets is used")
    parser.add_argument("--merge-datasets", action="store_true", help="This will merge the datasets into 3 single files (train.txt, test.txt, val.txt). Needs to be run after importing datasets")
    parser.add_argument("--dataset-config-path", type=str, default="datasets.json", help="Path to the dataset configuration file")
    parser.add_argument("--dataset-path", type=str, default="data", help="Path to the dataset directory")
    parser.add_argument("--dont-move-tokenizer", action="store_true", help="Don't move the tokenizer to the model directory")
    parser.add_argument("--eval-prompt", type=str, default="Once upon a time", help="The prompt to use for evaluation")
    parser.add_argument("--eval-max-new-tokens", type=int, default=50, help="The maximum number of new tokens to generate for evaluation")
    parser.add_argument("--resume-training", type=str, help="Resume training from a checkpoint. Can be: last, path to the directory checkpoints, or number of the epoch")
    parser.add_argument("--resume-from-epoch", type=int, help="Resume training from a specific epoch. If not specified, the script will try to ectract the epoch from the checkpoint name. If it can be found, it will resume from first epoch, so you can modify the config.json to reduce the epochs.")
    parser.add_argument("--train-seed", type=int, default=42, help="The seed to use for training")
    
    args = parser.parse_args()
    
        
    log_file = f"logs/{args.model_name}.log"
    if args.reset:
        if os.path.exists(log_file):
            os.remove(log_file)
            logging.info(f"Log file {log_file} deleted")
        if os.path.exists(args.model_name):
            shutil.rmtree(args.model_name, ignore_errors=True)
            logging.info(f"Model {args.model_name} deleted")
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    EVAL_PROMPT = args.eval_prompt
    EVAL_MAX_NEW_TOKENS = args.eval_max_new_tokens
    
    if not args.tokenizer_path and not args.generate_tokenizer:
        raise ValueError("Please provide a tokenizer path with --tokenizer-path or generate one with --generate-tokenizer")
    
    if args.generate_tokenizer:
        import dataset_manager
        import train_tokenizer
        dataset_manager.import_datasets(args.dataset_path)
        
        train_tokenizer.train_tokenizer(args.tokenizer_path, VOCAB_SIZE)
        
    if args.import_datasets:
        dataset_manager.import_datasets(args.dataset_config_path)
    
    if args.merge_datasets:
        if not dataset_manager.data_instances:
            dataset_manager.import_datasets(args.dataset_config_path)
        dataset_manager.merge_and_write_datasets(args.dataset_path)
        
    if not args.dont_move_tokenizer:
        if os.path.exists(f"{args.model_name}/tokenizer.json"):
            logging.info(f"Tokenizer already exists at {args.model_name}/tokenizer.json")
            if args.tokenizer_path != f"{args.model_name}/tokenizer.json":
                logging.warning(f"The tokenizer path ({args.tokenizer_path}) is different from the expected path ({args.model_name}/tokenizer.json). We'll use the specified tokenizer's path. Make sure you have the correct tokenizer.")
        else:
            logging.info(f"Moving tokenizer to {args.model_name}/tokenizer.json from {args.tokenizer_path} ...")
            os.move(args.tokenizer_path, f"{args.model_name}/tokenizer.json")
            logging.info(f"Tokenizer moved to {args.model_name}/tokenizer.json")
            args.tokenizer_path = f"{args.model_name}/tokenizer.json"
        
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    tokenizer_vocab_size = tokenizer.get_vocab_size()
    if tokenizer_vocab_size != VOCAB_SIZE:
        logging.warning(f"The tokenizer vocab size ({tokenizer_vocab_size}) is different from the expected vocab size ({VOCAB_SIZE}). We'll use the tokenizer's vocab size.")
        VOCAB_SIZE = tokenizer_vocab_size
        
LOG_EVERY = 500

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

if not args.resume_training:
    model = myTransformer(EMBED_DIM, NUM_HEADS, SEQ_LEN, VOCAB_SIZE, NUM_LAYERS, MLP_RATIO, DROPOUT)
    model.to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler(device=device)
    criterion = nn.CrossEntropyLoss() 
    
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader)*EPOCHS, eta_min=1e-5)

else:
    if args.resume_training == "last":
        checkpoints = os.listdir(f"{args.model_name}/checkpoints")
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {args.model_name}/checkpoints")
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        resume_checkpoint = f"{args.model_name}/checkpoints/{latest_checkpoint}"
        logging.info(f"Resuming training from {resume_checkpoint}")
    elif args.resume_training.isdigit():
        resume_checkpoint = f"{args.model_name}/checkpoints/{args.resume_training}"
        if not os.path.exists(resume_checkpoint):
            raise ValueError(f"Checkpoint {resume_checkpoint} not found")
        logging.info(f"Resuming training from {resume_checkpoint}")
    elif os.path.isdir(args.resume_training):
        resume_checkpoint = args.resume_training
    else:
        raise ValueError(f"Invalid resume_training argument: {args.resume_training}. Must be 'last', the number of the last checkpoint, or a directory containing a checkpoint.")
    
    if not os.path.exists(f"{resume_checkpoint}/optimizer.pt") or not os.path.exists(f"{resume_checkpoint}/scheduler.pt") or not os.path.exists(f"{resume_checkpoint}/model.pt"):
        raise ValueError(f"Checkpoints not found in {resume_checkpoint}. Make sure you have run the training script before and want to resume from checkpoints. Need: optimizer.pt, scheduler.pt, model.pt")
    model = myTransformer(EMBED_DIM, NUM_HEADS, SEQ_LEN, VOCAB_SIZE, NUM_LAYERS, MLP_RATIO, DROPOUT)
    model.load_state_dict(torch.load(f"{resume_checkpoint}/model.pt", map_location=device))
    optimizer = torch.load(f"{resume_checkpoint}/optimizer.pt", map_location=device)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min((step + 1) / 2000, 1.0))  # warmup on 2000 steps
    scheduler.load_state_dict(torch.load(f"{resume_checkpoint}/scheduler.pt", map_location=device))
    logging.info(f"Checkpoints loaded from {resume_checkpoint}")
    logging.info(f"Resuming training from {resume_checkpoint}")
    
    if args.resume_from_epoch:
        EPOCHS = EPOCHS - args.resume_from_epoch + 1
        logging.info(f"Resuming training from epoch {args.resume_from_epoch}. EPOCHS set to {EPOCHS + args.resume_from_epoch - 1} so stay {EPOCHS} epochs to train")
    else:
        epoch = int(resume_checkpoint.split("_")[-1].split(".")[0])
        if not epoch:
            logging.warning(f"Could not extract epoch from checkpoint name {resume_checkpoint}. Will resume from first epoch. Not bad but can produce overfitting.")
        else:
            EPOCHS = EPOCHS - epoch + 1
            logging.info(f"Resuming training from epoch {epoch}. EPOCHS set to {EPOCHS - epoch + 1} so stay {EPOCHS} epochs to train")
        
def save_checkpoint(model, optimizer, scheduler, epoch):
    save_dir = f"{args.model_name}/checkpoints/"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{save_dir}/model.pt")
    torch.save(optimizer.state_dict(), f"{save_dir}/optimizer.pt")
    torch.save(scheduler.state_dict(), f"{save_dir}/scheduler.pt")
    logging.info(f"Checkpoint saved: {save_dir}/{args.model_name}_epoch{epoch}.pt")
    
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, save_every=2):
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for step, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")):
            max_y = int(y.max().item())
            if max_y >= VOCAB_SIZE or y.min().item() < 0:
                tqdm.write(f"[ERROR] target id out of range: max {max_y} >= vocab_size {VOCAB_SIZE}")
                file_logger.error(f"Target id out of range: max {max_y} >= vocab_size {VOCAB_SIZE}")
                continue
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device): # fp16
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                
                
            if not torch.isfinite(loss):
                tqdm.write(f"[WARN] non-finite loss at step {step}: {loss.item()}")
                file_logger.warning(f"Non-finite loss at step {step}: {loss.item()}")
                scaler.update()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            try:
                scaler.step(optimizer)
            except Exception as e:
                tqdm.write(f"[ERROR] scaler.step(optimizer) failed: {e}")
                file_logger.error(f"scaler.step(optimizer) failed: {e}")
                scaler.update()
                continue
            scaler.update()
            scheduler.step()

            running_loss += float(loss.detach().item())

            if (step + 1) % LOG_EVERY == 0:
                avg_loss = running_loss / LOG_EVERY
                tqdm.write(f"[Epoch {epoch}/{epochs}] Step {step+1} | Train loss: {avg_loss:.4f}")
                file_logger.info(f"[Epoch {epoch}/{epochs}] Step {step+1} | Train loss: {avg_loss:.4f}")
                running_loss = 0.0
                
            if (step + 1) % (LOG_EVERY*4) == 0 or step == 0:
                generated = evaluate_prompt(model, tokenizer, EVAL_PROMPT, device, max_new_tokens=EVAL_MAX_NEW_TOKENS)
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
            save_checkpoint(model, optimizer, scheduler, epoch)
            logging.info(f"✅ New best model saved")

        # Sauvegarde périodique
        if epoch % save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch)
            logging.info(f"💾 Checkpoint saved")
                
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
    if len(ids) > SEQ_LEN:
        ids = ids[-SEQ_LEN:] 
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
