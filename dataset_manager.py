import json
import logging
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
            
        path = element.get("path", None)
        split = element.get("split", None)
        weight = float(element.get("weight", 1.0))
        data = Data(element["name"], element["config"], element["type"], path=path, split=split, weight=weight)
        data_instances.append(data)
        
        data.load()
        


def iter_txt_for_tokenizer():
    """Yield text for tokenizer training"""
    for data in data_instances:
        for text in data.extract_text_for_tokenizer():
            yield text

def merge_and_write_datasets(output_dir, apply_weights=True):
    """
    Merge all datasets with optional weighting.
    Weights > 1.0 duplicate data, < 1.0 subsample it.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name in ["train", "test", "val"]:
        output_path = os.path.join(output_dir, f"{split_name}.txt")
        
        with open(output_path, "w", encoding="utf-8") as f:
            for data_instance in tqdm(data_instances, desc=f"Writing {split_name}"):
                if not data_instance.loaded:
                    data_instance.load()
                
                lines = list(data_instance.get_lines(split_name))
                
                # Apply weight
                if apply_weights and data_instance.weight != 1.0:
                    if data_instance.weight > 1.0:
                        # Duplicate data (e.g., weight=2.0 → double the data)
                        repetitions = int(data_instance.weight)
                        lines = lines * repetitions
                        logging.info(f"⚖️  Duplicating {data_instance.name} x{repetitions} (weight={data_instance.weight})")
                    
                    elif data_instance.weight < 1.0:
                        # Subsample data (e.g., weight=0.5 → keep 50%)
                        import random
                        sample_size = int(len(lines) * data_instance.weight)
                        lines = random.sample(lines, sample_size)
                        logging.info(f"⚖️  Subsampling {data_instance.name} to {sample_size} lines (weight={data_instance.weight})")
                
                for line in lines:
                    f.write(line + "\n")
        
        logging.info(f"✅ {split_name}.txt written to {output_path}")
            
if __name__ == "__main__":
    import_datasets()
    merge_and_write_datasets()