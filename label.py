from util.label_safe_unsafe_pw import cosine_similarity, all_lower,all_upper, all_digit
from transformers import AutoTokenizer
import json
import yaml
from pathlib import Path
import random
import os
import pandas as pd



if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model_name=config["eval"]["config"]["model_name"]
    local_model_path = Path("model") / model_name
    model_source = str(local_model_path) if local_model_path.exists() else model_name
    tokenizer=AutoTokenizer.from_pretrained(model_source)

    train_dataset_path=Path(config["dataset_path"]) / "split" / "train_data.jsonl"
    test_dataset_path=Path(config["dataset_path"]) / "split" / "test_data.jsonl"
    output_path=Path(config["label"]["output_path"])

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    

    with open(train_dataset_path, "r") as f:
        len_train = sum(1 for line in f if line.strip())
    
    with open(test_dataset_path, "r") as f:
        len_test = sum(1 for line in f if line.strip())

    k = int(len_train * config["label"]["expected_ratio"] + 1)

    # 第二段：Reservoir sampling（Algorithm R），記憶體只保留 k 筆
    random.seed(config["label"]["seed"])
    reservoir = []
    with open(train_dataset_path, "r") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            if i < k:
                reservoir.append(item)
            else:
                j = random.randint(0, i)
                if j < k:
                    reservoir[j] = item
    train_select = reservoir

    output_item={
        "password": [],
        "safe_label":[]
    }
    for item in train_select:
        account=item.get("account", "")
        password=item.get("passwords","")

        consine_sim = cosine_similarity(account, password, tokenizer)
        if consine_sim > config["label"]["safe_threshold"] or all_lower(password) or all_upper(password) or all_digit(password):
            label=config["label"]["labels"]["unsafe"]
        else:
            label=config["label"]["labels"]["safe"]
        
        output_item["password"].append(password)
        output_item["safe_label"].append(label)

    pd.DataFrame(output_item).to_csv(output_path / "train_data.csv", index=False)
    
    k= int(len_test * config["label"]["expected_ratio"] + 1)
    reservoir = []
    with open(test_dataset_path, "r") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            if i < k:
                reservoir.append(item)
            else:
                j = random.randint(0, i)
                if j < k:
                    reservoir[j] = item
    test_select = reservoir
    output_item={
        "password": [],
        "safe_label":[]
    }
    for item in test_select:
        account=item.get("account", "")
        password=item.get("passwords","")

        consine_sim = cosine_similarity(account, password, tokenizer)
        if consine_sim > config["label"]["safe_threshold"] or all_lower(password) or all_upper(password) or all_digit(password):
            label=config["label"]["labels"]["unsafe"]
        else:
            label=config["label"]["labels"]["safe"]
        
        output_item["password"].append(password)
        output_item["safe_label"].append(label)

    pd.DataFrame(output_item).to_csv(output_path / "test_data.csv", index=False)

        



    