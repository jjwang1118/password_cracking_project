# 定義安全和不安全密碼
    #帳號和密碼間的余嫌相似度
    # 0.5以下為安全密碼，0.5以上為不安全密碼(暫定)
    # 純數字(過往研究)
    # 純小寫字母或純大寫字母(過往研究)
from transformers import AutoTokenizer
import json
from pathlib import Path
def cosine_similarity(account:str,password:str,tokenizer,gram:int=2):
    
    sos = tokenizer.bos_token or ""
    eos = tokenizer.eos_token or ""

    account_set=set()
    password_set=set()
    
    for idx in range(0, len(account) - gram + 1):
        account_set.add(account[idx:idx+gram])

    for idx in range(0, len(password) - gram + 1):
        password_set.add(password[idx:idx+gram])
    total_set=account_set.union(password_set)

    account_list= count(account, total_set)
    password_list = count(password, total_set)
    upper_value=0
    lower_account=0
    lower_password=0
    result=None
    for idx in range(len(total_set)):
       upper_value+=account_list[idx]*password_list[idx]
    
    for idx in range(len(total_set)):
        lower_account+=(account_list[idx]**2)
        lower_password+=(password_list[idx]**2)
    
    if lower_account == 0 or lower_password == 0:
        return 0.0
    else:
        result=upper_value/((lower_account**0.5)*(lower_password**0.5))
    return result
    

def count(str1: str, sets: set):
    sets_list = list(sets)
    return [str1.count(s) for s in sets_list]

def all_lower(s: str) -> bool:
    return s.isalpha() and s.islower()

def all_upper(s: str) -> bool:
    return s.isalpha() and s.isupper()

def all_digit(s: str) -> bool:
    return s.isdigit()

