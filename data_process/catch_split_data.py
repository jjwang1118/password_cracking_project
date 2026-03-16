import pandas as pd
import os
import json
from pathlib import Path
import random
import datasets
from datasets import load_dataset



def download_config():
    with open("config.json","r") as f:
        config=json.load(f)
    return config


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

    stastic_path=f"{Path.home()}/{dataset_path}/stastic_summary.json"
    with open (stastic_path,"r") as f:
        total_data=json.load(f)
    
    expected_nums=int(total_data["total_num"]*expected_ratio)
    catchdata_per_file=int(expected_nums/file_num)+1
    print(f"總共檔案數量: {file_num} , 每個檔案要抓取的資料數量: {catchdata_per_file}")


    for root, dirs,files in os.walk(outside_dataset_path):
        for file in files:
            idx_set=set()
            file_path=os.path.join(root,file)
            print(f"正在處理的檔案路徑: {file_path} \n")
            with open(os.path.join(root,file),"r",encoding="utf-8-sig",errors="ignore") as f:
                data=json.load(f)
                # 是jsonl檔案
                for item in range(len(data)):
                    #先抓取姊妹密碼數量大於1的資料
                    if data[item]["count"]>1:
                        idx_set.add(item)
                    
                    remain_num=catchdata_per_file-len(idx_set)
                    if remain_num<=0:
                        break
                    else:
                        while len(idx_set) !=catchdata_per_file:
                            #利用set的特性，確保不會抓取重複的資料
                            random_idx=random.sample(range(len(data)),remain_num)
                            idx_set.update(random_idx)

                            remain_num=catchdata_per_file-len(idx_set)
                
                for idx in idx_set:
                    item=data[idx]
                    account=item["account"] #str
                    passwords=item["password"] #list
                    with open(f"{Path.home()}/{dataset_path}/split/catch_data.jsonl","a",encoding="utf-8-sig") as f:
                        for password in passwords:
                            f.write(json.dumps({"account": account, "password": password}, ensure_ascii=False) + "\n")
        # 先全部載入到dataset中，之後再進行切分
        dataset=load_dataset("json", data_files=f"{Path.home()}/{dataset_path}/split/catch_data.jsonl")
        return dataset

def split_train_test(dataset, dataset_path, split_ratio: float, seed: int):
    # 傳入格式是jsonl
    dataset = dataset.shuffle(seed=seed) 
    split_data = dataset.train_test_split(test_size=split_ratio, seed=seed)

    with open(f"{Path.home()}/{dataset_path}/split/train_data.jsonl", "w", encoding="utf-8-sig") as f:
        for item in split_data["train"]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(f"{Path.home()}/{dataset_path}/split/test_data.jsonl", "w", encoding="utf-8-sig") as f:
        for item in split_data["test"]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")




