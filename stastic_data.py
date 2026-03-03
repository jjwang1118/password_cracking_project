import json
import os 
import pandas as pd
from pathlib import Path
from data_process.path_config import config 
from data_process.stastic import cac_not_valid_password_num
from data_process.stastic import org_domain_analysis
from data_process.stastic import len_analysis
from data_process.stastic import pair_sister
import gc

if __name__ == "__main__":
    cfg = config()
    stastic_path = cfg['stastic_path']
    total_orign, total_filter, toal_not_deplicate, toal_removed = cac_not_valid_password_num(stastic_path)

    with open('/home/jjwang1118/projects/password_cracking_project/datasets_filtered/stastic_summary.json', 'w', encoding='utf-8') as f:
        json.dump({
            "total_original_passwords": int(total_orign),
            "total_filtered_passwords": int(total_filter),
            "total_non_duplicate_passwords": int(toal_not_deplicate),
            "total_removed_passwords": int(toal_removed)
        }, f, ensure_ascii=False, indent=4)
    
    with open( Path.home() / "projects" / "password_cracking_project" / "data_process" / "task_finished.json", 'r', encoding='utf-8') as f:
        task_finished_data = json.load(f)
    
    print("Starting statistics analysis number of valid passwords\n")
    if task_finished_data['valid_password_num']['finished']:
        print("Already finished, skip.")
    else:
        cac_not_valid_password_num(stastic_path)
        task_finished_data['valid_password_num']['finished'] = True
        with open( Path.home() / "projects" / "password_cracking_project" / "data_process" / "task_finished.json", 'w', encoding='utf-8') as f:
            json.dump(task_finished_data, f, ensure_ascii=False, indent=4)
    print("Finished statistics analysis number of valid passwords\n")
    gc.collect()

    print("Starting statistics analysis of original domain and organization\n")
    if task_finished_data['org_domain_analysis']['finished']:
        print("Already finished, skip.")
    else:
        org_domain_analysis()
        task_finished_data['org_domain_analysis']['finished'] = True
        with open( Path.home() / "projects" / "password_cracking_project" / "data_process" / "task_finished.json", 'w', encoding='utf-8') as f:
            json.dump(task_finished_data, f, ensure_ascii=False, indent=4)
    print("Finished statistics analysis of original domain and organization\n")
    gc.collect()

    print("Starting statistics analysis of password length distribution\n")
    if task_finished_data['len_distribution_analysis']['finished']:
        print("Already finished, skip.")
    else:
        len_analysis()
        task_finished_data['len_distribution_analysis']['finished'] = True
        with open( Path.home() / "projects" / "password_cracking_project" / "data_process" / "task_finished.json", 'w', encoding='utf-8') as f:
            json.dump(task_finished_data, f, ensure_ascii=False, indent=4)
    print("Finished statistics analysis of password length distribution\n")
    gc.collect()

    print("Starting statistics analysis of sister password pairs\n")
    if task_finished_data['sister_password_analysis']['finished']:
        print("Already finished, skip.")
    else:
        pair_sister()
        task_finished_data['sister_password_analysis']['finished'] = True
        with open( Path.home() / "projects" / "password_cracking_project" / "data_process" / "task_finished.json", 'w', encoding='utf-8') as f:
            json.dump(task_finished_data, f, ensure_ascii=False, indent=4)
    print("Finished statistics analysis of sister password pairs\n")
    gc.collect()