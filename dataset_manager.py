import json
import os
from utils.data import Data
from tqdm import tqdm
data_instances = []
datasets = ""
def import_datasets(dataset_json_path = "datasets.json"):
    if not dataset_json_path:
        raise ValueError(f"Invalid dataset_json_path: {dataset_json_path}")
    
    with open(dataset_json_path, "r") as f:
        datasets = json.load(f)["datasets"]
        
    for element in datasets:
        if not "config" in element.keys():
            element["config"] = ""
    
        data = Data(element["name"], element["config"], element["type"])
        data_instances.append(data)
        
        data.load()
        


def iter_txt_for_tokenizer():
    """Yield text for tokenizer training"""
    for data in data_instances:
        for text in data.extract_text_for_tokenizer():
            yield text

def merge_and_write_datasets(dir_path="data"):
    os.makedirs(dir_path, exist_ok=True)
    splits = ["train", "test", "val"] 
    for split in tqdm(splits, desc="Merging datasets"):
        with open(f"data/{split}.txt", "w", encoding="utf-8") as fout:
            for d in data_instances:  # liste de Data
                for line in d.get_lines(split):
                    fout.write(line + "\n")
            
if __name__ == "__main__":
    import_datasets()
    merge_and_write_datasets()