import json
import re
import os
import pandas as pd


def is_ascii(password : str)->bool:
    if password.isascii():
        return True
    else:
        return False

def is_valid_password(password :str , password_map: str) -> bool:
    for char in password:
        if char not in password_map:
            return False
    return True

def is_valid_length(password:str,min_len:int, max_len:int) -> bool:
    if min_len <= len(password) <= max_len:
        return True
    else:
        return False

def remove_depulicates_passwords(account_password:pd.DataFrame):
    original_len = len(account_password)
    filtered_account_password = account_password.drop_duplicates(subset=['password','account'], keep='first')
    filtered_len = len(filtered_account_password)
    
    
    # 去重的帳密 ,原始長度 ,重複的數量
    return filtered_account_password, original_len, original_len - filtered_len

def merge_path(path1:str, path2:str) -> str:
    return os.path.join(path1, path2)


def data_process(config :dict , path : str , account_password:pd.DataFrame):
    
    account=[]
    pw=[]
    orignal_num=0
    filtered_num=0
    removed_depulicate_num=0
    with open(path, 'r', encoding='utf-8-sig', errors='ignore') as f:
        content=f.readlines()
        orignal_num=len(content)
        for line in content:
            account_pw=line.strip().split(":",-1)
            if len(account_pw) == 2 and is_ascii(account_pw[1]) and is_valid_password(account_pw[1], config['password_map']) and is_valid_length(account_pw[1], config['min_len'], config['max_len']):
                account.append(account_pw[0])
                pw.append(account_pw[1])
        f.close()
    account_password=pd.DataFrame({"account":account,"password":pw})
    account_password, filter_len,removed_num = remove_depulicates_passwords(account_password)

    # 資料 , 原始數量, 篩選後的數量 ,重複的數量
    return account_password, orignal_num,  filter_len, removed_num
    


    

    
    




