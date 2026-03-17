import os
import json
from pathlib import Path
from data_process.catch_split_data import (
    split_train_test,
    catch_data,
    count_password_length_distribution,
)
from datasets import load_dataset



if __name__ == "__main__":
    config_path = Path.home() / "projects" / "password_cracking_project" / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    
    seed = config["seed"]
    project_root = Path.home() / "projects" / "password_cracking_project"
    dataset_path = str(project_root / config["dataset_path"])

    if not os.path.exists(os.path.join(dataset_path,"train")) or not os.path.exists(os.path.join(dataset_path,"test")):
        os.makedirs(os.path.join(dataset_path,"split"), exist_ok=True)
    

    train_test_files = os.listdir(os.path.join(dataset_path,"split"))

    if len(train_test_files) == 2:
        print("train/test files already exist, skipping split.")
    else:
        print("Splitting data into train/test sets...")

        catch_data_path = os.path.join(dataset_path, "split", "catch_data.jsonl")
        if os.path.exists(catch_data_path) and os.path.getsize(catch_data_path) > 0:
            print("catch_data.jsonl already exists, skipping catch_data step.")
            train_test_dataset = load_dataset("json", data_files=catch_data_path)
        else:
            train_test_dataset = catch_data(config["expected_ratio"], dataset_path)

        split_train_test(train_test_dataset, dataset_path, config["split_ratio"], seed)


    if  not os.path.exists(os.path.join(dataset_path,"split","length_distribution.json")):
        print("Calculating length distribution...")
        length_distribution = {}
        length_distribution["train"] = count_password_length_distribution(
            f"{dataset_path}/split/train_data.jsonl"
        )
        length_distribution["test"] = count_password_length_distribution(
            f"{dataset_path}/split/test_data.jsonl"
        )
        
        with open(f"{dataset_path}/split/length_distribution.json", "w", encoding="utf-8-sig") as f:
            json.dump(length_distribution, f, ensure_ascii=False, indent=4)

