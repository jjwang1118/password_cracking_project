from torch.utils.data import Dataset  
import torch
import random
import pandas as pd
from pathlib import Path





class PasswordDataset(Dataset):
    # path 需要有完整路徑，包含檔名
    def __init__(self, data_path, tokenizer=None, max_length=64):
        self.data_path = data_path
        self.data = pd.read_csv(Path(data_path))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pw = str(self.data.iloc[idx]["password"])
        label = int(self.data.iloc[idx]["safe_label"])
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                pw,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
            )
            return encoded["input_ids"].squeeze(0), torch.tensor(label, dtype=torch.long)
        return pw, label
