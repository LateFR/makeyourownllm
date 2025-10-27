import json
import math
import os
import random
import re
import shutil
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from tokenizers import Tokenizer
import logging
import argparse
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import dataset_manager
logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="Name of the model")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer.json")
    parser.add_argument("--config-path", type=str, default="config.json")
    parser.add_argument("--reset", action="store_true", help="Reset the model current model and logs. To use if you had run the training script before and want to restart from scratch")
    parser.add_argument("--generate-tokenizer", action="store_true", help="Generate the tokenizer")
    parser.add_argument("--tokenizer-lang", type=str, default="en", choices=["fr", "en"], help="Language of the tokenizer. Don't support multilingual yet. Can be 'fr' or 'en'")
    parser.add_argument("--import-datasets", action="store_true", help="This will import the datasets from the datasets.json file, (downloading them from hungging face if necessary). Automatically called if --generate-tokenizer or --merge-datasets is used")
    parser.add_argument("--merge-datasets", action="store_true", help="This will merge the datasets into 3 single files (train.txt, test.txt, val.txt). Needs to be run to train the model")
    parser.add_argument("--dataset-config-path", type=str, default="datasets.json", help="Path to the dataset configuration file")
    parser.add_argument("--dataset-path", type=str, default="cleans_datasets_path.json", help="Path to the json file containing the dataset paths")
    parser.add_argument("--dont-move-tokenizer", action="store_true", help="Don't move the tokenizer to the model directory")
    parser.add_argument("--eval-prompt", type=str, default="Once upon a time", help="The prompt to use for evaluation")
    parser.add_argument("--eval-max-new-tokens", type=int, default=50, help="The maximum number of new tokens to generate for evaluation")
    parser.add_argument("--resume-training", type=str, help="Resume training from a checkpoint. Can be: last, path to the directory checkpoints, or number of the epoch")
    parser.add_argument("--resume-from-epoch", type=int, help="Resume training from a specific epoch. If not specified, the script will try to ectract the epoch from the checkpoint name. If it can be found, it will resume from first epoch, so you can modify the config.json to reduce the epochs.")
    parser.add_argument("--train-seed", type=int, default=42, help="The seed to use for training")
    parser.add_argument("--workers", type=int, default=0, help="The number of workers to use for the streaming dataset")
    args = parser.parse_args()
    
    
    log_file = f"logs/{args.model_name}.log"
    logging.FileHandler(log_file)
    if args.reset:
        if os.path.exists(log_file):
            os.remove(log_file)
            logging.info(f"Log file {log_file} deleted")
        if os.path.exists(args.model_name):
            shutil.rmtree(args.model_name, ignore_errors=True)
            logging.info(f"Model {args.model_name} deleted")
    os.makedirs("logs", exist_ok=True)
    os.makedirs(args.model_name, exist_ok=True)
    
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

    
    datasets_path = json.load(open(args.dataset_path, "r"))
    if "train" not in datasets_path.keys() or "test" not in datasets_path.keys() or "val" not in datasets_path.keys():
        raise ValueError(f"Invalid datasets_path: {args.dataset_path}. Must be a json file with the keys 'train', 'test', and 'val'")
        
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
    
    seed = args.train_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    logging.info(f"Training seed set to {seed}")
    
    if not args.tokenizer_path and not args.generate_tokenizer:
        raise ValueError("Please provide a tokenizer path with --tokenizer-path or generate one with --generate-tokenizer")
    
    if args.generate_tokenizer:
        import train_tokenizer
        dataset_manager.import_datasets(args.dataset_config_path)
        
        train_tokenizer.train_tokenizer(args.tokenizer_path, VOCAB_SIZE, args.tokenizer_lang)
        
    if args.import_datasets:
        dataset_manager.import_datasets(args.dataset_config_path)
    
    if args.merge_datasets:
        if not dataset_manager.data_instances:
            dataset_manager.import_datasets(args.dataset_config_path)
        dataset_manager.merge_and_write_datasets("data")
        
    if not args.dont_move_tokenizer:
        if os.path.exists(f"{args.model_name}/tokenizer.json"):
            logging.info(f"Tokenizer already exists at {args.model_name}/tokenizer.json")
            if args.tokenizer_path != f"{args.model_name}/tokenizer.json":
                logging.warning(f"The tokenizer path ({args.tokenizer_path}) is different from the expected path ({args.model_name}/tokenizer.json). We'll use the specified tokenizer's path. Make sure you have the correct tokenizer.")
        else:
            logging.info(f"Copying tokenizer to {args.model_name}/tokenizer.json from {args.tokenizer_path} ...")
            shutil.copy(args.tokenizer_path, f"{args.model_name}/tokenizer.json")
            logging.info(f"Tokenizer moved to {args.model_name}/tokenizer.json")
            args.tokenizer_path = f"{args.model_name}/tokenizer.json"
        
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    tokenizer_vocab_size = tokenizer.get_vocab_size()
    if tokenizer_vocab_size != VOCAB_SIZE:
        logging.warning(f"The tokenizer vocab size ({tokenizer_vocab_size}) is different from the expected vocab size ({VOCAB_SIZE}). We'll use the tokenizer's vocab size.")
        VOCAB_SIZE = tokenizer_vocab_size
    
    warmup_steps = config.get("warmup_steps", 2000)
    total_steps = config.get("total_steps", None)
    lr_scale_min = config.get("lr_scale_min", 0.1)
    best_val_loss = float("inf")
    global_step = 0

    
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
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=50, top_p=0.95):
        self.eval()
        vocab_size = self.token_embedding.num_embeddings
        block_size = self.pos_embedding.num_embeddings
        
        for _ in range(max_new_tokens):
            context = input_ids if input_ids.size(1) <= block_size else input_ids[:, -block_size:]
            
            logits = self(context)[:, -1, :] / temperature
            
            # Check for NaN/Inf in the logits
            if not torch.isfinite(logits).all():
                break
                
            probs = torch.softmax(logits, dim=-1)
            
            # top-k
            if top_k is not None and top_k > 0:
                top_k_clamped = min(top_k, vocab_size)
                values, _ = torch.topk(probs, top_k_clamped)
                min_values = values[:, -1].unsqueeze(1)
                probs = torch.where(probs < min_values, torch.zeros_like(probs), probs)
                
                # CRITIQUE : Check that the sum is not null
                prob_sum = probs.sum(dim=-1, keepdim=True)
                if (prob_sum < 1e-10).any():
                    # Fallback : reset to a uniform distribution over top-k
                    probs = torch.where(probs > 0, torch.ones_like(probs), torch.zeros_like(probs))
                    prob_sum = probs.sum(dim=-1, keepdim=True)
                probs = probs / prob_sum
            
            # top-p
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_probs[cum_probs > top_p] = 0.0
                
                # CRITICAL: Check that the sum is not null
                prob_sum = sorted_probs.sum(dim=-1, keepdim=True) + 1e-12
                if (prob_sum < 1e-10).any():
                    sorted_probs[:, 0] = 1.0
                    prob_sum = sorted_probs.sum(dim=-1, keepdim=True)
                
                sorted_probs = sorted_probs / prob_sum
                probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)
            
            # Last check before sampling
            if not torch.isfinite(probs).all() or (probs.sum(dim=-1) < 1e-10).any():
                break
            
            next_token = torch.multinomial(probs, num_samples=1)
            next_token = torch.clamp(next_token, 0, vocab_size - 1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


def make_worker_init_fn(base_seed):
    def _worker_init_fn(worker_id):
        seed = base_seed + worker_id
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    return _worker_init_fn

    
class StreamingDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, tokenizer, block_size):
        self.path = path
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        start_line = 0
        step = 1
        if worker_info is not None:
            # découper les lignes par worker
            step = worker_info.num_workers
            start_line = worker_info.id
        with open(self.path, "r", encoding="utf-8") as f:
            buffer = []
            for idx, line in enumerate(f):
                if (idx - start_line) % step != 0:
                    continue
                tokens = self.tokenizer.encode(line.strip()).ids
                buffer.extend(tokens)
                while len(buffer) >= self.block_size + 1:
                    x = torch.tensor(buffer[:self.block_size], dtype=torch.long)
                    y = torch.tensor(buffer[1:self.block_size+1], dtype=torch.long)
                    yield x, y
                    buffer = buffer[1:]


GLOBAL_WORKER_SEED = None  # set at the bottom

def worker_init_fn(worker_id):
    if GLOBAL_WORKER_SEED is not None:
        seed = GLOBAL_WORKER_SEED + worker_id
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
  
def make_lr_lambda(warmup_steps=2000, total_steps=None, lr_scale_min=0.1):
    if total_steps is None:
        total_steps = warmup_steps + 200_000  # fallback (large)

    def _lambda(step):
        # step is 0-based number of scheduler.step() calls
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = min(float(step - warmup_steps) / float(max(1, total_steps - warmup_steps)), 1.0)
        # cosine decay from 1.0 -> lr_scale_min
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_scale_min + (1.0 - lr_scale_min) * cosine_decay

    return _lambda  
def save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_val_loss):
    save_dir = os.path.join(args.model_name, "checkpoints", str(epoch))
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{save_dir}/model.pt")
    torch.save(optimizer.state_dict(), f"{save_dir}/optimizer.pt")
    torch.save(scheduler.state_dict(), f"{save_dir}/scheduler.pt")
    meta = {"epoch": epoch, "global_step": global_step, "best_val_loss": best_val_loss}
    with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    logging.info(f"Checkpoint saved: {save_dir}")
    
def load_checkpoint_for_resume(resume_checkpoint, model, optimizer, scheduler, device):
    # resume_checkpoint is path to epoch_{N} directory
    model.load_state_dict(torch.load(os.path.join(resume_checkpoint, "model.pt"), map_location=device))
    optimizer.load_state_dict(torch.load(os.path.join(resume_checkpoint, "optimizer.pt"), map_location=device))
    scheduler.load_state_dict(torch.load(os.path.join(resume_checkpoint, "scheduler.pt"), map_location=device))
    meta_path = os.path.join(resume_checkpoint, "meta.json")
    meta = {"epoch": 0, "global_step": 0, "best_val_loss": float("inf")}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return meta

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, global_step=0, best_val_loss=float("inf"), save_every=2):
    
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

            global_step += 1
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
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_val_loss)
            logging.info(f"✅ New best model saved")

        # Sauvegarde périodique
        if epoch % save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_val_loss)
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

    result = model.generate(input_ids, max_new_tokens=max_new_tokens)
    decoded = tokenizer.decode(result[0].tolist(), skip_special_tokens=True)

    # Nettoyage simple / sûr pour Metaspace markers
    s = decoded.replace('▁', ' ')          # remplace le marker Metaspace par un espace
    s = re.sub(r'\s+([.,;:!?%)»»›])', r'\1', s)   # retire espace avant ponctuation courante
    s = re.sub(r'([(\[«‹])\s+', r'\1', s)         # retire espace après ouvrants éventuels
    s = re.sub(r'\s{2,}', ' ', s).strip()         # compact multiple espaces
    return s



if __name__ == "__main__":
    GLOBAL_WORKER_SEED = args.train_seed
    train_loader = DataLoader(
        StreamingDataset(datasets_path["train"], tokenizer, SEQ_LEN),
        batch_size=BATCH_SIZE,
        num_workers=args.workers,
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        StreamingDataset(datasets_path["val"], tokenizer, SEQ_LEN),
        batch_size=BATCH_SIZE,
        num_workers=args.workers,
        worker_init_fn=worker_init_fn
    )
    test_loader = DataLoader(
        StreamingDataset(datasets_path["test"], tokenizer, SEQ_LEN),
        batch_size=BATCH_SIZE,
        num_workers=args.workers,
        worker_init_fn=worker_init_fn
    )

    
    scaler = GradScaler(device=device)
    criterion = nn.CrossEntropyLoss() 
    
    if not args.resume_training:
        model = myTransformer(EMBED_DIM, NUM_HEADS, SEQ_LEN, VOCAB_SIZE, NUM_LAYERS, MLP_RATIO, DROPOUT)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=make_lr_lambda(warmup_steps, total_steps, lr_scale_min))

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
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=make_lr_lambda(warmup_steps, total_steps, lr_scale_min))
        meta = load_checkpoint_for_resume(resume_checkpoint, model, optimizer, scheduler, device)
        global_step = meta.get("global_step", 0)
        best_val_loss = meta.get("best_val_loss", float("inf"))
        epoch = meta.get("epoch", 0)
        EPOCHS = EPOCHS - epoch + 1
        logging.info(f"Resuming training from epoch {epoch}. EPOCHS set to {EPOCHS - epoch + 1} so stay {EPOCHS} epochs to train")
        
        logging.info(f"Checkpoints loaded from {resume_checkpoint}")
        logging.info(f"Resuming training from {resume_checkpoint}")
    logging.info(f"Vocab size: {tokenizer.get_vocab_size()}")
    print(f"Max token ID possible: {max(tokenizer.get_vocab().values())}")
    logging.info("Starting training...")
    os.makedirs(f"{args.model_name}", exist_ok=True)
    with open(f"{args.model_name}/config.json", "w") as f:
        json.dump(config, f)
    train_model(model, train_loader, val_loader, optimizer, criterion, EPOCHS, global_step, best_val_loss)
    logging.info("Evaluating on test set...")
    evaluate(model, test_loader, criterion, "Test")
    os.makedirs(f"{args.model_name}/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"{args.model_name}/checkpoints/{args.model_name}.pt")
    logging.info("Training complete.")
