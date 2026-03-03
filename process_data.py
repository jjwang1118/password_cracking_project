# goal
# 刪除非法字元
# 刪除非password map的資料
# 保留長度 >8
# 刪除重複地的資料
# 統計清洗前後資料
# 統計 domain 
# 統計長度分布
# 紀錄目前處理到哪



import os
import csv
from pathlib import Path
import pandas as pd
from data_process.path_config import config
from data_process.process_function import data_process
from data_process.process_function import merge_path


if __name__ == "__main__":

    config=config()
    dataset_path=config['leak_dataset_path']
    output_path=config['output_path']
    stastic_path = Path.home() / "projects" / "password_cracking_project" / "datasets_filtered" / "stastic.csv"
    # check and create output path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # read data
    for root ,dir ,file in os.walk(dataset_path):
        for f in file:
            file_path=merge_path(root,f)
            print(f"正在處理的檔案路徑: {file_path} \n")
            
            # 檢查是否已處理過
            need_write_new = True
            if os.path.exists(stastic_path) and os.path.getsize(stastic_path) > 0:
                try:
                    history_record = pd.read_csv(stastic_path)
                    matched = history_record[history_record['path'] == file_path]
                    if not matched.empty:
                        if matched.iloc[0]['finished'] == True:
                            print(f"{file_path} already processed, skip.")
                            continue
                        else:
                            # 已有紀錄但未完成，不需要再寫入新 record
                            print(f"{file_path} found incomplete record, resuming...")
                            need_write_new = False
                except pd.errors.EmptyDataError:
                    pass  # 檔案空的，當作不存在處理

            # 統計資料和旗標
            record={
                "path":file_path,
                "orignal_num":0,
                "filter_num":0, # 不包含去除重複的動作
                "removed_num":0,
                "finished":False #去除重複的數量
            }
            
            # 只有全新的檔案才寫入新紀錄
            if need_write_new:
                file_exists = os.path.exists(stastic_path)
                with open(stastic_path, 'a', encoding='utf-8-sig', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=record.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(record)
            
            account_pw=[]
            account=password=[]
            with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                content=f.readlines()
                for line in content:
                    line=line.strip()
                    account_pw.append(line)
            
            for line in account_pw:
                data=line.strip().split(":",-1)
                for c in data[0]:
                    if not c.isascii():
                        continue
                account.append(data[0])
                password.append(data[1])
            
            account_password=pd.DataFrame({"account":account,"password":password})
            account_password, orignal_num, filter_len, removed_num = data_process(config, file_path, account_password)

            # 輸出處理後的資料，保持原目錄結構
            relative_path = os.path.relpath(file_path, dataset_path)
            output_file = os.path.join(output_path, relative_path)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            account_password.to_csv(output_file, index=False, encoding='utf-8-sig')

            record['orignal_num']=orignal_num
            record['filter_num']=filter_len
            record['removed_num']=removed_num

            # record - 用 path 定位並更新該筆紀錄
            try:
                df = pd.read_csv(stastic_path)
                matched_idx = df[df['path'] == file_path].index
                if len(matched_idx) > 0:
                    idx = matched_idx[0]
                    df.loc[idx, 'orignal_num'] = orignal_num
                    df.loc[idx, 'filter_num'] = filter_len
                    df.loc[idx, 'removed_num'] = removed_num
                    df.loc[idx, 'finished'] = True
                    df.to_csv(stastic_path, index=False, encoding='utf-8-sig')
                else:
                    print(f"警告：找不到 {file_path} 的紀錄，無法更新")
            except Exception as e:
                print(f"警告：更新紀錄失敗 - {e}")

            print(f"檔案 {file_path} 處理完成，原始數量: {orignal_num}, 篩選後的數量: {filter_len}, 去除重複的數量: {removed_num} \n")
