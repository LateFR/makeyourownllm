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
    def __init__(self, name, config, type):
        self.name = name
        self.config = config
        self.type = type
        
        self.train = []
        self.test = []
        self.val = []
        
        self.loaded = False
        
        
        
    def load(self):
        dataset = load_dataset(self.name, self.config)
        
        if "train" in dataset.keys():
            self.train.append(dataset["train"])
        if "test" in dataset.keys():
            self.test.append(dataset["test"])
        if "validation" in dataset.keys():
            self.val.append(dataset["validation"])
            
        tqdm(f"{self.name} loaded")
        self.loaded = True
        tqdm.write(f"{self.name} total size: {self.get_size()}")
    
    def get_size(self):
        size = 0
        if not self.loaded:
            raise ValueError("Data not loaded")
        
        for dataset_split in self.train + self.test + self.val:
            for x in dataset_split:
                for k in x.keys():
                    try:
                        size += len(x[k])
                    except TypeError:
                        pass
            
        return size
    
    def get_lines(self, split_name):
        if split_name not in ["train", "test", "val"]:
            return []

        split_datasets = (
            self.train if split_name == "train" else
            self.test if split_name == "test" else
            self.val
        )

        for dataset in split_datasets:  # <- chaque dataset = datasets.Dataset
            for x in dataset:  # <- chaque x = une ligne = dict
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
                                yield " <SEP> ".join([t.strip() for t in v if isinstance(t, str) and t.strip()]) + " <EOS>"
                        except Exception as e:
                            logger.error(f"Error extracting text: {e}")

    def extract_text_for_tokenizer(self):
        texts = []
        for dataset_split in self.train + self.test + self.val:
            for x in dataset_split:
                # texte simple
                if self.type == "text" and "text" in x:
                    if isinstance(x["text"], str) and x["text"].strip():
                        texts.append(x["text"].strip() + " <EOS>")
                # dialogues
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

            