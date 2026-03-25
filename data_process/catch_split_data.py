import os
import json
import yaml
import random
from datasets import DatasetDict, load_dataset



def download_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


def count_password_length_distribution(jsonl_path):
    distribution = {}
    with open(jsonl_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            password = item.get("password")
            if password is None:
                # Backward compatibility for files that may still use "passwords" key.
                password = item.get("passwords", "")

            length = len(password)
            if length in distribution:
                distribution[length] += 1
            else:
                distribution[length] = 1

    return distribution


def _select_target_indices(file_path, target_count):
    if target_count <= 0:
        return set()

    priority_indices = []

    # Pass 1: keep earliest records where count > 1, matching original priority behavior.
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        data_idx = 0
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            if item.get("count", 0) > 1 and len(priority_indices) < target_count:
                priority_indices.append(data_idx)
            data_idx += 1

    remain = target_count - len(priority_indices)
    if remain <= 0:
        return set(priority_indices)

    # Pass 2: reservoir sample from non-priority records.
    reservoir = []
    seen = 0
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        data_idx = 0
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            if item.get("count", 0) > 1:
                data_idx += 1
                continue

            seen += 1
            if len(reservoir) < remain:
                reservoir.append(data_idx)
            else:
                replace_idx = random.randint(0, seen - 1)
                if replace_idx < remain:
                    reservoir[replace_idx] = data_idx
            data_idx += 1

    selected = set(priority_indices)
    selected.update(reservoir)
    return selected


def catch_data(expected_ratio : int,dataset_path):

    config=download_config()
    random.seed(config["seed"])
    outside_dataset_path=None
    # 抓取總共檔案比數
    for root,dirs,files in os.walk(dataset_path):
        for dir in dirs:
            if "sister" in dir:
                outside_dataset_path=os.path.join(dataset_path,dir)
                for sub_root,sub_dirs,sub_files in os.walk(outside_dataset_path):
                    file_num=len(sub_files)
    
    # 沒筆資料要抓多少

    stastic_path=f"{dataset_path}/stastic_summary.json"
    with open (stastic_path,"r") as f:
        total_data=json.load(f)
    
    expected_nums=int(total_data["total_filtered_passwords"]*expected_ratio)
    catchdata_per_file=int(expected_nums/file_num)+1
    print(f"總共檔案數量: {file_num} , 每個檔案要抓取的資料數量: {catchdata_per_file}")

    output_path = f"{dataset_path}/split/catch_data.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        pass


    for root, dirs,files in os.walk(outside_dataset_path):
        for file in files:
            file_path=os.path.join(root,file)
            print(f"正在處理的檔案路徑: {file_path} \n")
            selected_indices = _select_target_indices(file_path, catchdata_per_file)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as src, open(
                output_path, "a", encoding="utf-8"
            ) as out:
                data_idx = 0
                for line in src:
                    line = line.strip()
                    if not line:
                        continue

                    if data_idx in selected_indices:
                        item = json.loads(line)
                        account = item.get("account", "")
                        passwords = item.get("passwords", [])

                        if isinstance(passwords, list):
                            password_iter = passwords
                        else:
                            password_iter = [passwords]

                        for password in password_iter:
                            out.write(
                                json.dumps(
                                    {"account": account, "passwords": password},
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                    data_idx += 1
        # 先全部載入到dataset中，之後再進行切分
        dataset=load_dataset("json", data_files=output_path)
        return dataset

def split_train_test(dataset, dataset_path, split_ratio: float, seed: int):
    # load_dataset("json", ...) returns DatasetDict with a default "train" split.
    if isinstance(dataset, DatasetDict):
        if "train" not in dataset:
            raise ValueError("DatasetDict does not contain a 'train' split.")
        dataset = dataset["train"]

    dataset = dataset.shuffle(seed=seed)
    split_data = dataset.train_test_split(test_size=split_ratio, seed=seed)

    with open(f"{dataset_path}/split/train_data.jsonl", "w", encoding="utf-8") as f:
        for item in split_data["train"]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(f"{dataset_path}/split/test_data.jsonl", "w", encoding="utf-8") as f:
        for item in split_data["test"]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")




