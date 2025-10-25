import argparse
import dataset_manager
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=100000)
    
    args = parser.parse_args()

    dataset_manager.load_datasets()
    
txt_for_tokenizer = dataset_manager.iter_txt_for_tokenizer()

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(
    vocab_size=args.vocab_size,
    show_progress=True,
    special_tokens=["<PAD>", "<BOS>", "<EOS>", "<SEP>", "<UNK>"]
)
tokenizer.pre_tokenizer = Whitespace()
tokenizer.train_from_iterator(txt_for_tokenizer, trainer)
tokenizer.save(args.tokenizer_path)
logging.info(f"Tokenizer saved at {args.tokenizer_path}")
