import os
from datasets import load_dataset
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

class Data:
    def __init__(self, name, config=None, type="text", path=None, weight=1.0, split=None):
        self.name = name
        self.config = config
        self.type = type
        self.path = path
        self.weight = weight
        self.split = split  # ← Format: "train[:1000]" ou "train[:10%]"
        
        self.train = []
        self.test = []
        self.val = []
        
        self.loaded = False
    
    def _parse_split_notation(self, split_str):
        """
        Parse les notations de split HuggingFace:
        - "train[:1000]" → prendre les 1000 premiers
        - "train[:10%]" → prendre les 10% premiers
        - "train[1000:2000]" → prendre de 1000 à 2000
        - "train" → prendre tout
        
        Retourne (split_name, slice_obj)
        """
        if not split_str:
            return None, None
        
        # Format: "train[:1000]" ou "train[1000:2000]"
        if '[' in split_str and ']' in split_str:
            split_name = split_str[:split_str.index('[')]
            slice_part = split_str[split_str.index('[')+1:split_str.index(']')]
            
            # Parse le slice
            if ':' in slice_part:
                parts = slice_part.split(':')
                start = parts[0].strip()
                end = parts[1].strip() if len(parts) > 1 else None
                
                # Gérer les pourcentages
                if end and '%' in end:
                    return split_name, ('percent', None, int(end.replace('%', '')))
                elif start and '%' in start:
                    return split_name, ('percent', int(start.replace('%', '')), None)
                else:
                    start_int = int(start) if start else None
                    end_int = int(end) if end else None
                    return split_name, ('absolute', start_int, end_int)
            else:
                # Format: "train[1000]" → prendre jusqu'à 1000
                if '%' in slice_part:
                    return split_name, ('percent', None, int(slice_part.replace('%', '')))
                else:
                    return split_name, ('absolute', None, int(slice_part))
        else:
            # Pas de slice, prendre tout
            return split_str, None
    
    def _apply_slice(self, dataset, slice_info):
        """Applique un slice à un dataset HuggingFace."""
        if not slice_info:
            return dataset
        
        slice_type, start, end = slice_info
        total = len(dataset)
        
        if slice_type == 'percent':
            if start is not None:
                start_idx = int(total * start / 100)
            else:
                start_idx = 0
            
            if end is not None:
                end_idx = int(total * end / 100)
            else:
                end_idx = total
            
            logger.info(f"  → Applying percentage slice: {start or 0}% to {end or 100}% = {start_idx} to {end_idx} items")
            return dataset.select(range(start_idx, min(end_idx, total)))
        
        elif slice_type == 'absolute':
            if start is None:
                start = 0
            if end is None:
                end = total
            
            logger.info(f"  → Applying absolute slice: {start} to {end} items")
            return dataset.select(range(start, min(end, total)))
        
        return dataset
    
    def load(self):
        if self.path:
            # Chargement depuis fichier local
            if not self.path.endswith(".txt"):
                raise ValueError(f"Invalid path: {self.path} must be a .txt file")
            if not os.path.exists(self.path):
                raise ValueError(f"File not found: {self.path}")
            
            with open(self.path, "r", encoding="utf-8") as f:
                all_lines = [line.strip() for line in f if line.strip()]
            
            total = len(all_lines)
            
            # Split: 80% train, 10% val, 10% test
            train_end = int(total * 0.8)
            val_end = int(total * 0.9)
            
            self.train = [all_lines[:train_end]]
            self.val = [all_lines[train_end:val_end]]
            self.test = [all_lines[val_end:]]
            
            logger.info(f"Loaded {self.name} from file: {len(self.train[0])} train, {len(self.val[0])} val, {len(self.test[0])} test")
        
        else:
            # Chargement depuis HuggingFace
            if self.split:
                # Parse le split personnalisé
                split_name, slice_info = self._parse_split_notation(self.split)
                
                logger.info(f"Loading {self.name} with custom split: {self.split}")
                dataset = load_dataset(self.name, self.config, split=split_name)
                
                # Appliquer le slice
                dataset = self._apply_slice(dataset, slice_info)
                
                # Tout mettre dans train par défaut (puisque c'est un subset custom)
                self.train.append(dataset)
                logger.info(f"  → Loaded {len(dataset)} items into train split")
            
            else:
                # Chargement standard (tous les splits)
                dataset = load_dataset(self.name, self.config)
                
                if "train" in dataset.keys():
                    self.train.append(dataset["train"])
                if "test" in dataset.keys():
                    self.test.append(dataset["test"])
                if "validation" in dataset.keys():
                    self.val.append(dataset["validation"])
        
        self.loaded = True
        logger.info(f"{self.name} total size: {self.get_size()}")
    
    def get_size(self):
        if not self.loaded:
            raise ValueError("Data not loaded")
        
        size = 0
        for dataset_split in self.train + self.test + self.val:
            # Custom data (list of strings)
            if isinstance(dataset_split, list):
                size += sum(len(line) for line in dataset_split)
            # HuggingFace dataset
            else:
                for x in dataset_split:
                    for k in x.keys():
                        try:
                            size += len(str(x[k]))
                        except TypeError:
                            pass
        
        return size
    
    def get_lines(self, split_name):
        if split_name not in ["train", "test", "val"]:
            return
        
        split_datasets = (
            self.train if split_name == "train" else
            self.test if split_name == "test" else
            self.val
        )
        
        for dataset in split_datasets:
            # Custom data (plain text list)
            if isinstance(dataset, list):
                for line in dataset:
                    if isinstance(line, str) and line.strip():
                        yield line.strip() + " <EOS>"
            
            # HuggingFace dataset
            else:
                for x in dataset:
                    if self.type == "text" and "text" in x:
                        text = x["text"].strip()
                        if text:
                            yield text + " <EOS>"
                    elif self.type == "dialog" and "dialog" in x:
                        if isinstance(x["dialog"], list) and x["dialog"]:
                            dialogue = " <SEP> ".join([t.strip() for t in x["dialog"] if t.strip()])
                            yield dialogue + " <EOS>"
                    else:
                        for k, v in x.items():
                            try:
                                if isinstance(v, str) and v.strip():
                                    yield v.strip() + " <EOS>"
                                elif isinstance(v, list) and v:
                                    yield " <SEP> ".join([str(t).strip() for t in v if isinstance(t, str) and t.strip()]) + " <EOS>"
                            except Exception as e:
                                logger.error(f"Error extracting text: {e}")
    
    def extract_text_for_tokenizer(self):
        texts = []
        
        for dataset_split in self.train + self.test + self.val:
            # Custom data (plain text list)
            if isinstance(dataset_split, list):
                for line in dataset_split:
                    if isinstance(line, str) and line.strip():
                        texts.append(line.strip() + " <EOS>")
            
            # HuggingFace dataset
            else:
                for x in dataset_split:
                    if self.type == "text" and "text" in x:
                        if isinstance(x["text"], str) and x["text"].strip():
                            texts.append(x["text"].strip() + " <EOS>")
                    elif self.type == "dialog" and "dialog" in x:
                        if isinstance(x["dialog"], list) and x["dialog"]:
                            dialogue = " <SEP> ".join([str(t).strip() for t in x["dialog"] if str(t).strip()])
                            texts.append(dialogue + " <EOS>")
                    else:
                        for k in x.keys():
                            try:
                                if isinstance(x[k], str) and x[k].strip():
                                    texts.append(x[k].strip() + " <EOS>")
                                elif isinstance(x[k], list) and x[k]:
                                    texts.append(" <SEP> ".join([str(t).strip() for t in x[k] if str(t).strip()]) + " <EOS>")
                            except TypeError:
                                pass
                            except Exception as e:
                                logger.error(f"Error while extracting text for tokenizer: {e}")
        
        return texts