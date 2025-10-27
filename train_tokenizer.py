import argparse
from typing import Literal
import dataset_manager
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace, Whitespace
from tokenizers.normalizers import NFD, NFC
from tokenizers import normalizers
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def train_tokenizer(tokenizer_path, vocab_size, lang:Literal["fr", "en"]="en"):
    txt_for_tokenizer = dataset_manager.iter_txt_for_tokenizer()

    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=["<PAD>", "<BOS>", "<EOS>", "<SEP>", "<UNK>"]
    )
    logging.info(f"Training tokenizer for language: {lang}")
    if lang == "fr":
        tokenizer.pre_tokenizer = Metaspace()
        tokenizer.normalizer = normalizers.Sequence([
            NFD(),
            NFC()
        ])
    else:
        tokenizer.pre_tokenizer = Whitespace()
        
    tokenizer.train_from_iterator(txt_for_tokenizer, trainer)
    if lang == "fr":
        tokenizer.decoder = Metaspace()
    else:
        tokenizer.decoder = Whitespace()
    tokenizer.save(tokenizer_path)
    logging.info(f"Tokenizer saved at {tokenizer_path}")
    # CHECK POST-TRAINING
    logging.info(f"Tokenizer saved at {tokenizer_path}")
    logging.info(f"Vocab size: {tokenizer.get_vocab_size()}")
    max_id = max(tokenizer.get_vocab().values())
    logging.info(f"Max token ID: {max_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=100000)
    parser.add_argument("--lang", type=str, default="en", choices=["fr", "en"])
    
    args = parser.parse_args()

    dataset_manager.load_datasets()
    train_tokenizer(args.tokenizer_path, args.vocab_size, args.lang)
