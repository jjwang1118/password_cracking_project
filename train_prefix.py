from util.generate_control import llamaModel_control
from util.datacollector import PasswordDataset
import torch
from transformers import AutoModelForCausalLM
import yaml

if __name__ == "__main__":
    # 載入參數
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    label_config = config["label"]
    model_name = label_config["model_name"]
    model_path = f"model/{model_name}"

    # 載入預訓練模型，從中取得 hf_config
    base_model = AutoModelForCausalLM.from_pretrained(model_path)
    hf_config = base_model.config

    # 初始化自訂模型（prefix 參數隨機初始化）
    model = llamaModel_control(hf_config, label_config)

    # 載入預訓練權重（strict=False 忽略 parameterlist）
    model.load_state_dict(base_model.state_dict(), strict=False)
    del base_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 載入訓練資料
    train_dataset = PasswordDataset(label_config["train_test_dataset"]["train_path"])

    # 開始訓練
    model.step(train_dataset)
