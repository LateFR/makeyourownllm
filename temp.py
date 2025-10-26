import os
import sys
from tokenizers import Tokenizer
from tqdm import tqdm

def count_tokens_in_file(file_path, tokenizer, chunk_size=8192):
    total_tokens = 0
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            tokens = tokenizer.encode(chunk)
            total_tokens += len(tokens)
    return total_tokens

def count_tokens_in_folder(folder_path, model_name="tokenizer"):
    tokenizer = Tokenizer.from_file(f"{model_name}.json")
    total_tokens = 0

    # Récupère la liste des fichiers à traiter
    files_to_process = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.txt', '.json', '.jsonl')):
                files_to_process.append(os.path.join(root, file))

    # Boucle avec barre de progression
    for path in tqdm(files_to_process, desc="🧮 Comptage des tokens", ncols=90):
        total_tokens += count_tokens_in_file(path, tokenizer)

    return total_tokens

def format_number(n):
    return f"{n:,}".replace(",", " ")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_tokens.py <chemin_du_dataset> [tokenizer_name]")
        sys.exit(1)

    folder = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "tokenizer"

    print(f"🔍 Comptage des tokens dans : {folder}")
    print(f"🧠 Tokenizer utilisé : {model}")

    total = count_tokens_in_folder(folder, model)
    print(f"\n✅ Total tokens : {format_number(total)}")

    # Estimation : 4 bytes par float32 * nombre de tokens pour un embedding
    estimated_memory = total * 4 / (1024 ** 3)
    print(f"🧠 Estimation mémoire float32 : {estimated_memory:.2f} Go")
