from torch.nn.data import Dataset
import json
import random
import pandas as pd
from pathlib import Path





class PasswordDataset(Dataset):
    # path 需要有完整路徑，包含檔名
    def __init__(self,data_path):
        self.data_kind=data_kind
        self.data_path=data_path

        dataset_path=Path(self.data_path)
        self.data=pd.read_csv(dataset_path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data.iloc[idx]["password"], self.data.iloc[idx]["safe_label"]
