# 修正 1：移除無效的 Optimizer import；修正 DataLoader 路徑 torch.nn.data → torch.utils.data
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from transformers import get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader
from util.datacollector import PasswordDataset
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


def hiddenhead(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config.get("num_attention_heads", "")


def hidden_dim(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config.get("head_dim", "")


def model_layers(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config.get("num_hidden_layers", "")


class llamaModel_control(LlamaForCausalLM):
    #加入 hf_config 參數並呼叫 super().__init__(hf_config)，否則父類未初始化
    def __init__(self, hf_config, label_config: dict):
        super().__init__(hf_config)
        self.hidden_head = hf_config.num_attention_heads
        self.per_head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        self.layers = hf_config.num_hidden_layers
        self.label_config = label_config
        self.prefix_length = self.label_config.get("prefix_len", 10)
        self.dropout = torch.nn.Dropout(self.label_config.get("dropout", 0.2))
        self.batch_size = self.label_config.get("batch_size", 4)
        self.lr = self.label_config.get("learning_rate", 1e-4)
        self.weight_decay = self.label_config.get("weight_decay", 0.01)
        self.warmup_ratio = self.label_config.get("warmup_ratio", 0.1)
        self.grad_acc = self.label_config.get("grad_acc", 2)
        self.parameterlist = torch.nn.ParameterList()
        for _ in range(2):          # sec / not sec
            for _ in range(self.layers):
                for _ in range(2):  # k / v
                    param = torch.nn.Parameter(
                        torch.zeros(self.hidden_head, self.prefix_length, self.per_head_dim)
                    )
                    self.parameterlist.append(param)

    def _load_config(self):
        with open("config.json", "r") as f:
            return json.load(f)

    def load_model(self, model_path):
        # self 本身就是模型，只需凍結非 prefix 的參數
        for name, param in self.named_parameters():
            param.requires_grad = name.startswith("parameterlist")
        return self

    def download_data(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def step(self, dataset):
        config = self._load_config()
        eff_batch = self.batch_size * self.grad_acc
        total_steps = (len(dataset) // eff_batch) * config["label"].get("epochs", 1)

        self.load_model(Path("model") / config["label"]["model_name"])
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        writer = SummaryWriter(log_dir=Path("logs") / config["label"]["model_name"])

        checkpoint_list = {
            "optimizer": None,
            "scheduler": None,
            "parameterlist": None,
            "train_loss": 0.0,
            "avg_train_loss": 0.0,
            "avg_eval_loss": 0.0,
            "epochs": []
        }

        for epoch in range(config["label"].get("epochs", 1)):
            checkpoint_list, avg_train_loss = self.train_epoch(
                optimizer, scheduler, config, dataset, epoch, checkpoint_list
            )
            checkpoint_list, avg_eval_loss = self.eval_epoch(
                config, dataset, epoch, checkpoint_list
            )
            writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
            writer.add_scalar("Loss/Eval", avg_eval_loss, epoch + 1)

    def train_epoch(self, optimizer, scheduler, config, dataset, epoch, checkpoint_list):
        self.train()
        checkpoint_list["train_loss"] = 0.0
        print(f"Epoch {epoch + 1}/{config['label'].get('epochs', 1)}\n")

  

        for idx, (pw, control_id) in enumerate(
            self.download_data(PasswordDataset(config["label"]["train_test_dataset"]["train_path"]))
        ):
            device = next(self.parameters()).device
            pw = pw.to(device)
            control_id = control_id.to(device)

            loss = torch.tensor(0.0, device=device)

            _, output = self(pw, control_id=control_id)
            loss = loss + self.loss_function("cross_entropy", output, pw) * self.label_config.get("loss", {}).get("lm_loss", 1)

            # 錯誤 control（翻轉：0→1, 1→0）的 forward（contrastive loss）
            wrong_control_id = 1 - control_id
            _, wrong_output = self(pw, control_id=wrong_control_id)
            loss = loss + self.loss_function("nll", wrong_output, pw) * self.label_config.get("loss", {}).get("consta_loss", 1)

            # 無 prefix 的 forward（KL loss，保持非密碼相關生成行為不變）
            self.eval()
            with torch.no_grad():
                _, ref_output = self(pw, control_id=None)
            self.train()
            loss = loss + self.loss_function("kl", output, ref_output) * self.label_config.get("loss", {}).get("kl_loss", 1)

            if (idx + 1) % self.grad_acc == 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            checkpoint_list["train_loss"] += loss.item()
            print(f"Step {idx + 1}/{len(dataset) // self.batch_size}, Loss: {loss.item():.4f}")

        # 每個 epoch 結束都存 checkpoint（訓練總 epoch 數少，參考 SVEN save_epochs=1）
        checkpoint_list["avg_train_loss"] = checkpoint_list["train_loss"] / max(len(dataset), 1)
        checkpoint_list["optimizer"] = optimizer.state_dict()
        checkpoint_list["scheduler"] = scheduler.state_dict()
        checkpoint_list["parameterlist"] = self.parameterlist.state_dict()
        checkpoint_list["epochs"].append({"epoch": epoch + 1, "avg_train_loss": checkpoint_list["avg_train_loss"]})
        torch.save(
            checkpoint_list,
            Path("checkpoints") / config["label"]["model_name"] / f"checkpoint_epoch_{epoch + 1}.pt"
        )
        return checkpoint_list, checkpoint_list["avg_train_loss"]

    def eval_epoch(self, config, dataset, epoch, checkpoint_list):
        self.eval()
        eval_loss = 0.0
        # 修正 8：torch,no_grad() 逗號應為點，改為 torch.no_grad()
        with torch.no_grad():
            for idx, (pw, control_id) in enumerate(
                self.download_data(PasswordDataset(config["label"]["train_test_dataset"]["test_path"]))
            ):
                device = next(self.parameters()).device
                pw = pw.to(device)
                control_id = control_id.to(device)
                loss = torch.tensor(0.0, device=device)
                _, output = self(pw, control_id=control_id)
                loss = loss + self.loss_function("cross_entropy", output, pw) * self.label_config.get("loss", {}).get("lm_loss", 1)
                eval_loss += loss.item()
                print(f"Eval Step {idx + 1}/{len(dataset) // self.batch_size}, Eval Loss: {loss.item():.4f}")

        checkpoint_list["avg_eval_loss"] = eval_loss / max(len(dataset), 1)
        if checkpoint_list["epochs"]:
            checkpoint_list["epochs"][-1]["avg_eval_loss"] = checkpoint_list["avg_eval_loss"]
        return checkpoint_list, checkpoint_list["avg_eval_loss"]

    def loss_function(self, loss_type, inputs, targets):
        # 原始程式碼有 weight 參數，本專案不需要，已刪除
        if loss_type == 'cross_entropy':
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        elif loss_type == 'nll':
            loss_fct = torch.nn.NLLLoss(reduction='none')
        elif loss_type == 'kl':
            loss_fct = torch.nn.KLDivLoss(log_target=True, reduction='none')

        loss = loss_fct(inputs, targets)
        if loss_type == 'kl':
            loss = loss.sum(dim=1)
        return loss.mean()

    def get_past_from_prefix(self, control_ids):
        """將 prefix_params 轉換為 past_key_values 格式"""
        """
            建kv cache
            past_key_values = [
                (key_layer_0, val_layer_0),   # shape: (batch, heads, prefix_len, head_dim)
                (key_layer_1, val_layer_1),
                ...
            ]       
        """
        past = []
        control_ids = [int(c.item()) if isinstance(c, torch.Tensor) else c for c in control_ids]
        for i in range(self.layers):
            key_stack, val_stack = [], []
            for control_id in control_ids:
                key_idx = control_id * self.layers * 2 + i * 2
                val_idx = key_idx + 1
                key_stack.append(self.dropout(self.parameterlist[key_idx]))
                val_stack.append(self.dropout(self.parameterlist[val_idx]))
            past.append((torch.stack(key_stack), torch.stack(val_stack)))
        return past

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        """生成時自動注入 prefix"""
        if not past:
            control_ids = [kwargs['control_id']] * input_ids.shape[0]
            past = self.get_past_from_prefix(control_ids)
        else:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
        }



