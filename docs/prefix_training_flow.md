# Prefix-Tuning 完整訓練流程

## 目錄

1. [核心思想](#1-核心思想)
2. [啟動訓練](#2-啟動訓練)
3. [模型載入與凍結](#3-模型載入與凍結)
4. [資料集建構](#4-資料集建構)
5. [訓練迴圈詳解](#5-訓練迴圈詳解)
6. [單步訓練 step(batch) 詳解](#6-單步訓練-stepbatch-詳解)
7. [模型 Forward Pass 詳解](#7-模型-forward-pass-詳解)
8. [三個 Loss 函數](#8-三個-loss-函數)
9. [參數更新機制](#9-參數更新機制)
10. [Prefix 與 LLM 的關係](#10-prefix-與-llm-的關係)
11. [驗證與保存](#11-驗證與保存)
12. [超參數配置](#12-超參數配置)
13. [訓練 vs 推理](#13-訓練-vs-推理)
14. [完整流程總覽](#14-完整流程總覽)

---

## 1. 核心思想

本專案（SVEN）的目標是：**在不改動預訓練 LLM（如 CodeGen）權重的情況下**，透過學習 **prefix 向量**（一組可訓練的 key/value 向量），引導模型生成「安全」或「有漏洞」的程式碼。

每個 `control_id` 對應一組獨立的 prefix 參數：
- `control_id = 0`：secure（安全）
- `control_id = 1`：vulnerable（漏洞）

Prefix 的作用方式是在每一層 Transformer 的 attention 機制中，拼接額外的 Key/Value 向量，使模型在計算 attention 時「看到」額外的虛擬 prefix token，從而改變生成行為。

---

## 2. 啟動訓練

### 2.1 執行命令

```bash
python scripts/train.py --output_name 2b-prefix --pretrain_dir 2b
```

### 2.2 進入點（`scripts/train.py`）

```python
def main():
    args = get_args()
    set_logging(args, os.path.join(args.output_dir, 'train.log'))
    set_devices(args)
    set_seed(args)

    if args.model_type == 'prefix':
        trainer = PrefixTrainer(args)
    elif args.model_type == 'text':
        trainer = TextPromptTrainer(args)

    trainer.run()  # ← 啟動訓練
```

`trainer.run()` 定義在 `TrainerBase` 中，依序呼叫：
1. `load_model()` — 載入模型並凍結
2. `load_dataset()` — 載入訓練/驗證資料
3. 訓練迴圈 — 執行訓練

### 2.3 自動超參數設定

在 `get_args()` 中根據模型大小自動配置：

| 模型 | `n_prefix_token` | `epochs` | `kl_loss_ratio` | `learning_rate` |
|------|:-:|:-:|:-:|:-:|
| 350M | 5 | 8 | 1600 | 1e-2 |
| 2B | 8 | 5 | 1600 | 1e-2 |
| 6B | 12 | 5 | 2000 | 1e-2 |

---

## 3. 模型載入與凍結

### 3.1 `PrefixTrainer.load_model()`

```python
def load_model(self):
    self.tokenizer, self.model, self.input_device = load_model('prefix', self.args.pretrain_dir, True, self.args)
    for n, p in self.model.named_parameters():
        if n.startswith('prefix_params'):
            p.requires_grad = True    # 只有 prefix 參數可訓練
        else:
            p.requires_grad = False   # LM 全部凍結
    self.model.train()
```

### 3.2 `load_model()` 內部（`sven/model.py`）

訓練模式下的載入流程：

```python
# is_training = True 時
lm_path = path  # e.g. 'Salesforce/codegen-2B-multi'
lm_config = config_from_pretrained(lm_path, lm_path)
lm_config.n_prefix_token = args.n_prefix_token  # e.g. 8
lm_config.prefix_dropout = args.dropout          # e.g. 0.1
lm_config.n_control = 2                          # secure + vulnerable
model = model_from_pretrained(lm_path, 'prefix', lm_config)
```

### 3.3 Prefix 參數結構

以 `CodeGenPrefixCausalLM` 為例（假設 CodeGen-2B, 32 層, 16 head, 每 head 128 維）：

```python
self.prefix_params = ParameterList()
for control in range(2):           # 0=secure, 1=vulnerable
    for layer in range(32):        # 模型的每一層
        for kv in range(2):        # Key 和 Value
            param = Parameter(zeros(16, 8, 128))  # (n_head, n_prefix_token, dim_per_head)
            self.prefix_params.append(param)
```

**參數索引對照表：**

| 索引 | Control | Layer | Key/Value |
|------|---------|-------|-----------|
| 0 | secure (0) | Layer 0 | Key |
| 1 | secure (0) | Layer 0 | Value |
| 2 | secure (0) | Layer 1 | Key |
| ... | ... | ... | ... |
| 62 | secure (0) | Layer 31 | Key |
| 63 | secure (0) | Layer 31 | Value |
| 64 | vulnerable (1) | Layer 0 | Key |
| 65 | vulnerable (1) | Layer 0 | Value |
| ... | ... | ... | ... |
| 127 | vulnerable (1) | Layer 31 | Value |

**參數量計算：**
- 單個參數張量：`16 × 8 × 128 = 16,384`
- 總 prefix 參數量：`2 × 32 × 2 × 16,384 = 2,097,152` ≈ 210 萬
- LLM 參數量 ≈ 20 億
- 可訓練比例 ≈ 0.1%

### 3.4 模型記憶體結構

```
model
├── transformer (凍結，不訓練)
│   ├── wte (word embedding)
│   ├── h.0 ~ h.31 (32 層 Transformer)
│   └── ln_f (final layer norm)
├── lm_head (凍結)
├── prefix_params[0]   ← secure, layer 0, Key    ✅ 可訓練
├── prefix_params[1]   ← secure, layer 0, Value   ✅ 可訓練
│   ...
├── prefix_params[63]  ← secure, layer 31, Value  ✅ 可訓練
├── prefix_params[64]  ← vul, layer 0, Key       ✅ 可訓練
│   ...
└── prefix_params[127] ← vul, layer 31, Value    ✅ 可訓練
```

---

## 4. 資料集建構

### 4.1 `PrefixDataset`

```python
def load_dataset(self):
    self.dataset = PrefixDataset(self.args, self.tokenizer, 'train')
    self.val_dataset = PrefixDataset(self.args, self.tokenizer, 'val')
```

### 4.2 資料來源

從 `data_train_val/train/` 讀取 `.jsonl` 檔案（如 `cwe-089.jsonl`）。每筆 JSON 資料包含：

```json
{
  "func_src_after": "def query(name):\n    cursor.execute('SELECT * FROM users WHERE name = ?', (name,))",
  "func_src_before": "def query(name):\n    cursor.execute('SELECT * FROM users WHERE name = ' + name)",
  "line_changes": {"added": [...], "deleted": [...]},
  "char_changes": {"added": [...], "deleted": [...]}
}
```

- `func_src_after` = 修復後的安全版本 → `control_id = 0` (secure)
- `func_src_before` = 修復前的漏洞版本 → `control_id = 1` (vulnerable)

**每筆 JSON 會產生 2 筆訓練資料（一安全、一漏洞）。**

### 4.3 單筆資料的產生（`get_tensor()`）

```python
def get_tensor(self, src, vul_id, control_id, changes):
    be = self.tokenizer.encode_plus(src)
    tokens = be.data['input_ids']
    if len(tokens) > self.args.max_num_tokens: return None

    if changes is None:
        weights = [1] * len(tokens)
    else:
        weights = [0] * len(tokens)
        for change in changes:
            char_start_idx = be.char_to_token(change['char_start'])
            char_end_idx = be.char_to_token(change['char_end'] - 1)
            for char_idx in range(char_start_idx, char_end_idx + 1):
                weights[char_idx] = 1

    return tokens, weights, control_id, vul_id
```

**回傳格式：**

| 欄位 | 型別 | 範例 | 說明 |
|------|------|------|------|
| `tokens` | List[int] | `[318, 12405, 7, ...]` | 原始碼的 token ids |
| `weights` | List[int] | `[0, 0, ..., 1, 1, ..., 0]` | 差異位置 = 1，其餘 = 0 |
| `control_id` | int | `0` 或 `1` | 0 = secure, 1 = vulnerable |
| `vul_id` | int | `0` | 漏洞類型索引 |

### 4.4 差異層級（`diff_level`）

| 參數值 | 說明 |
|--------|------|
| `prog` | 所有 token 權重 = 1（整段程式都參與 loss） |
| `line` | 僅在行級差異處權重 = 1 |
| `char` | 僅在字元級差異處權重 = 1 |
| `mix`（預設） | secure 用 char 差異、vulnerable 用 line 差異 |

---

## 5. 訓練迴圈詳解

### 5.1 初始化 Optimizer 與 Scheduler

```python
batch_size = 1
train_sampler = RandomSampler(self.dataset)
train_dataloader = DataLoader(self.dataset, sampler=train_sampler, batch_size=batch_size, drop_last=True)

total_samples = len(self.dataset)
batch_size = batch_size * self.args.grad_acc_steps  # 等效 batch_size = 2
total_steps = total_samples // batch_size * self.args.num_train_epochs

optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
```

### 5.2 主循環結構

```python
global_step, acc_loss_dict = 0, OrderedDict()
set_seed(self.args)
self.model.train()  # ← 設定訓練模式（啟用 dropout），不是開始訓練

for idx in range(self.args.num_train_epochs):          # 外層：Epoch 循環
    for step, batch in enumerate(train_dataloader):    # 內層：Batch 循環
        loss, loss_dict = self.step(batch)             # 計算 loss（3 次 forward pass）

        # grad_acc_steps > 1 代表有啟用梯度累積，需要縮放 loss
        # grad_acc_steps = 1 代表不累積，每步直接更新，loss / 1 無意義故跳過
        # 注意：梯度累積的真正原因是 .backward() 預設行為為 grad += 新梯度（累積而非覆蓋）
        #       只要不呼叫 zero_grad()，梯度就會一直累積
        #       這個 if 的作用只是把 loss 縮放，讓累積後的梯度等效於「取平均」而非「取總和」
        if self.args.grad_acc_steps > 1:
            loss = loss / self.args.grad_acc_steps     # 梯度累積縮放（等效 batch mean）
            for key in loss_dict:
                loss_dict[key] = loss_dict[key] / self.args.grad_acc_steps  # 同步縮放供 logging 用（不影響反向傳播）

        loss.backward()                                # 反向傳播，累積梯度
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)  # 梯度裁剪
        self.add_to_loss_dict(acc_loss_dict, loss_dict)  # 累積統計

        # parser.add_argument('--grad_acc_steps', type=int, default=2)
        if (step+1) % self.args.grad_acc_steps == 0:   # 每 2 步更新一次
            optimizer.step()                           # 更新 prefix_params
            optimizer.zero_grad()                      # 清空梯度
            scheduler.step()                           # 更新學習率
            global_step += 1

            if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                reported_loss = self.report_loss_dict(acc_loss_dict, self.args.logging_steps)
                self.args.logger.info(...)
                acc_loss_dict.clear()
```

### 5.3 `model.train()` 的作用

`model.train()` **不是**開始訓練，而是切換模型狀態：

| 模式 | 方法 | Dropout | 用途 |
|------|------|---------|------|
| 訓練模式 | `model.train()` | 啟用 | 訓練時使用 |
| 評估模式 | `model.eval()` | 停用 | 推理/驗證時使用 |

### 5.4 `enumerate(train_dataloader)` 說明

```python
for step, batch in enumerate(train_dataloader):
    # step: 當前 batch 的序號（每個 epoch 從 0 重新開始）
    # batch: 一筆訓練資料 (inputs, weights, control_ids, vul_ids)
```

- `DataLoader` 的 `batch_size=1`，每次取 1 筆資料
- `RandomSampler` 每個 epoch 隨機打亂資料順序
- `drop_last=True` 丟棄最後不足一個 batch 的資料
- `step` 用於控制梯度累積的時機：`(step+1) % grad_acc_steps == 0`

### 5.5 梯度累積邏輯

```
grad_acc_steps = 2 (每 2 個 batch 才更新一次參數)

Step 0: forward → backward（梯度累積到 prefix_params.grad，不清零）
        (0+1) % 2 = 1 ≠ 0 → 不更新

Step 1: forward → backward（梯度繼續累積，兩步的總和）
        (1+1) % 2 = 0 → 更新參數！清空梯度

Step 2: forward → backward（重新開始累積）
        (2+1) % 2 = 1 → 不更新

Step 3: forward → backward
        (3+1) % 2 = 0 → 更新參數！
...
```

**效果：等同於 `batch_size = 2`，但記憶體消耗更低。**

### 5.6 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
```

計算所有梯度的總範數，如果超過 `max_grad_norm = 1.0`，則等比例縮放所有梯度：

```python
total_norm = sqrt(sum(p.grad.norm(2)^2 for p in parameters))
if total_norm > 1.0:
    for p in parameters:
        p.grad *= 1.0 / total_norm
```

### 5.7 三種計數器

| 計數器 | 範圍 | 重置時機 | 用途 |
|--------|------|---------|------|
| `idx` | 0 ~ num_epochs-1 | 每次訓練 | 追蹤當前第幾個 epoch |
| `step` | 0 ~ len(dataset)-1 | 每個 epoch | 控制梯度累積時機 |
| `global_step` | 1 ~ total_steps | 從不重置 | 追蹤全域參數更新次數 |

### 5.8 完整時序範例

假設 `num_train_epochs=5`, `grad_acc_steps=2`, `logging_steps=100`, `save_epochs=1`：

```
【Epoch 0】(idx=0)
  Step 0, 1   → global_step=1 (更新參數)
  Step 2, 3   → global_step=2
  ...
  Step 198, 199 → global_step=100 → 記錄日誌
  ...
  Step 9998, 9999 → global_step=5000
  → 驗證集評估 → 保存 checkpoint-epoch-1

【Epoch 1】(idx=1)
  Step 0, 1   → global_step=5001  ← step 重置，global_step 繼續
  ...
  → 保存 checkpoint-epoch-2

【Epoch 2-4】
  同理...

【訓練結束】共 25000 次參數更新
```

---

## 6. 單步訓練 step(batch) 詳解

### 6.1 準備輸入

```python
def step(self, batch):
    inputs, weights, control_ids, _ = batch
    inputs = inputs.to(self.input_device)          # (1, seq_len) → GPU
    shift_inputs = inputs[..., 1:].squeeze(0)      # (seq_len-1,) 去掉第一個 token
    weights = weights.to(self.input_device)
    shift_weights = weights[..., 1:].squeeze(0)    # (seq_len-1,) 去掉第一個 weight
    control_ids = control_ids.to(self.input_device) # (1,) 值為 0 或 1
```

**shift 的原因：** 語言模型預測「下一個 token」，所以 logits 和 labels 要錯開一位。

```
inputs:       [318, 12405, 7, 3672, 1267, 889, 50, 100]
shift_inputs:      [12405, 7, 3672, 1267, 889, 50, 100]  ← 預測目標

logits 位置 0 預測 → 12405
logits 位置 1 預測 → 7
...
```

### 6.2 三次模型 Forward Pass

**每個 batch 會執行 3 次模型 forward pass：**

| 次數 | control_id | Prefix 參數索引 | 用途 |
|------|-----------|----------------|------|
| 1st | 正確的（e.g. 0） | [0-63] | 計算 lm_loss |
| 2nd | 相反的（e.g. 1） | [64-127] | 計算 contrastive_loss |
| 3rd | None（無 prefix） | 無 | 計算 kl_loss（參考分佈） |

### 6.3 `get_logits_from_lm()` 函數

```python
def get_logits_from_lm(lm, inputs, control_ids):
    if control_ids is not None:
        past = lm.get_past_from_prefix(control_ids)  # 從 prefix 參數生成 past_key_values
    else:
        past = None  # 不使用 prefix

    outputs = lm(inputs, past_key_values=past)  # 模型 forward
    shift_logits = outputs.logits[..., :-1, :]  # (1, seq_len-1, vocab_size)
    shift_labels = inputs[..., 1:].unsqueeze(-1)
    shift_probs = F.softmax(shift_logits, dim=-1)
    return shift_logits.squeeze(0), torch.gather(shift_probs, 2, shift_labels).squeeze(-1).squeeze(0)
```

**回傳：**
- `shift_logits`：`(seq_len-1, vocab_size)` — 每個位置對所有 token 的 logit
- `label_probs`：`(seq_len-1,)` — 每個位置對正確 token 的機率

---

## 7. 模型 Forward Pass 詳解

### 7.1 `get_past_from_prefix()` — 將 Prefix 轉為 past_key_values

```python
def get_past_from_prefix(self, control_ids):
    # control_ids = [0] (secure)
    past = []
    for i in range(self.config.n_layer):  # 遍歷 32 層
        key_idx = control_id * n_layer * 2 + i * 2
        val_idx = key_idx + 1
        key = self.dropout(self.prefix_params[key_idx])  # (16, 8, 128)
        val = self.dropout(self.prefix_params[val_idx])  # (16, 8, 128)
        past.append([stacked_keys, stacked_vals])
    return past
```

**control_id = 0 (secure) 時的索引：**
```
Layer 0: key_idx=0,  val_idx=1  → past[0]=[params[0], params[1]]
Layer 1: key_idx=2,  val_idx=3  → past[1]=[params[2], params[3]]
...
Layer 31: key_idx=62, val_idx=63 → past[31]=[params[62], params[63]]
```

**control_id = 1 (vulnerable) 時的索引：**
```
Layer 0: key_idx=64, val_idx=65 → past[0]=[params[64], params[65]]
...
```

### 7.2 `CodeGenPrefixCausalLM.forward()` — 透傳給父類

```python
def forward(self, input_ids, past_key_values=None, ..., control_id=None):
    # control_id 是佔位符，此函數內未使用
    # prefix 資訊已包含在 past_key_values 中
    return super().forward(
        input_ids=input_ids,
        past_key_values=past_key_values,  # ← prefix 以標準格式傳入
        ...
    )
```

### 7.3 Transformer 內部執行流程

```python
# === Embedding Layer ===
hidden = model.transformer.wte(input_ids)  # (1, 512) → (1, 512, 2048)

# === 逐層 Transformer (Layer 0 ~ 31) ===
for i, layer in enumerate(self.transformer.h):
    layer_past = past_key_values[i]  # [prefix_key, prefix_value]

    # 計算 Q, K, V
    Q = layer.attn.q_proj(hidden)  # (1, 512, 2048)
    K = layer.attn.k_proj(hidden)  # (1, 512, 2048)
    V = layer.attn.v_proj(hidden)  # (1, 512, 2048)

    # 分割成多頭
    Q = Q.view(1, 512, 16, 128).transpose(1, 2)  # (1, 16, 512, 128)
    K = K.view(1, 512, 16, 128).transpose(1, 2)  # (1, 16, 512, 128)
    V = V.view(1, 512, 16, 128).transpose(1, 2)  # (1, 16, 512, 128)

    # ★★★ 拼接 Prefix ★★★
    prefix_K, prefix_V = layer_past
    K = torch.cat([prefix_K, K], dim=2)  # (1, 16, 8+512, 128) = (1, 16, 520, 128)
    V = torch.cat([prefix_V, V], dim=2)  # (1, 16, 520, 128)

    # Attention 計算
    attn_scores = Q @ K.transpose(-1, -2) / sqrt(128)  # (1, 16, 512, 520)
    attn_weights = softmax(attn_scores, dim=-1)
    attn_output = attn_weights @ V                      # (1, 16, 512, 128)

    # FFN
    hidden = layer.mlp(attn_output)

# === Final ===
hidden = model.transformer.ln_f(hidden)  # (1, 512, 2048)
logits = model.lm_head(hidden)           # (1, 512, 51200)
```

**重點：每個 token 在做 attention 時，都會「看到」前面額外的 8 個 prefix token。Prefix 不改變 Q（query），只改變 K（key）和 V（value），從而影響 attention 的輸出。**

### 7.4 有/無 Prefix 的 Attention 對比

| 情境 | K, V 的序列長度 | 說明 |
|------|----------------|------|
| 無 Prefix | 512 | 原始 LLM 行為 |
| 有 Prefix | 8 + 512 = 520 | 每個 token 額外關注 8 個 prefix token |

---

## 8. 三個 Loss 函數

### 8.1 LM Loss — 「用正確的 prefix 預測程式碼」

```python
correct_logits, correct_label_probs = get_logits_from_lm(self.model, inputs, control_ids)
lm_loss = token_weighted_loss('cross_entropy', correct_logits, shift_inputs, shift_weights)
lm_loss *= self.args.lm_loss_ratio  # 預設 1
```

**計算過程：**
```
logits:  [...]  [...]  [...]  [...]  [...]  [...]  [...]
target: 12405    7    3672  1267   889    50   100
weights:   0     0     0     0     1     1     1
                                   ↑     ↑     ↑
                         只計算這些差異 token 的 cross-entropy
```

**意義：** 讓正確的 prefix 在「安全相關的差異 token」位置，引導模型生成正確的程式碼。

### 8.2 Contrastive Loss — 「正確 prefix 要比錯誤 prefix 更好」

```python
incorrect_control_ids = -1 * (control_ids - 1)  # 0→1, 1→0
incorrect_logits, incorrect_label_probs = get_logits_from_lm(self.model, inputs, incorrect_control_ids)

contrastive_probs = torch.stack((correct_label_probs, incorrect_label_probs), dim=1)
contrastive_probs = F.normalize(contrastive_probs, p=1, dim=-1)
contrastive_log_probs = torch.log(contrastive_probs)
contrastive_labels = torch.zeros(shift_inputs.shape, dtype=torch.int64)
contrastive_loss = token_weighted_loss('nll', contrastive_log_probs, contrastive_labels, shift_weights)
contrastive_loss *= self.args.contrastive_loss_ratio / 100  # 預設 400/100 = 4.0
```

**計算過程（差異 token 位置 i）：**
```
correct_prob[i]   = 0.85  ← secure prefix 生成正確 token 的機率
incorrect_prob[i] = 0.42  ← vulnerable prefix 生成正確 token 的機率

normalize → [0.669, 0.331]
NLL(label=0) = -log(0.669) = 0.40

意義：強迫 correct_prob > incorrect_prob
```

### 8.3 KL Loss — 「非差異部分不要改變」

```python
correct_log_probs = F.log_softmax(correct_logits, dim=-1)
self.model.eval()
with torch.no_grad():
    ref_logits, _ = get_logits_from_lm(self.model, inputs, None)  # ← 不帶 prefix
self.model.train()
ref_log_probs = F.log_softmax(ref_logits, dim=-1)

kl_loss += token_weighted_loss('kl', correct_log_probs, ref_log_probs, 1-shift_weights)
kl_loss += token_weighted_loss('kl', incorrect_log_probs, ref_log_probs, 1-shift_weights)
kl_loss = kl_loss * self.args.kl_loss_ratio / 1000  # 預設 1600/1000 = 1.6
```

**關鍵細節：**
- `control_ids=None` → 不使用任何 prefix，得到原始 LLM 的分佈作為參考
- `model.eval()` → 停用 dropout，確保得到乾淨的參考分佈
- `torch.no_grad()` → 不計算梯度（這些是參考值）
- `weights = 1 - shift_weights` → 在**非差異 token** 上計算 KL 散度

```
shift_weights:      [0, 0, 0, 0, 1, 1, 1, 0, 0]
1 - shift_weights:  [1, 1, 1, 1, 0, 0, 0, 1, 1]
                     ↑  ↑  ↑  ↑           ↑  ↑
                     只在這些「非差異」位置計算 KL 散度
```

**意義：** prefix 的存在不應該改變模型在「跟安全無關的程式碼」上的行為，保證功能正確性不受影響。

### 8.4 最終 Loss 合併

```python
loss = lm_loss + contrastive_loss + kl_loss
```

各 loss 的典型量級：
```
lm_loss          = 2.45 × 1 (lm_loss_ratio)       = 2.45
contrastive_loss = 0.42 × 4.0 (400/100)           = 1.68
kl_loss          = 0.007 × 1.6 (1600/1000)         = 0.0112
                                          Total    ≈ 4.14
```

### 8.5 `token_weighted_loss()` 通用加權 Loss

```python
def token_weighted_loss(loss_type, inputs, targets, weights):
    if loss_type == 'cross_entropy':
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    elif loss_type == 'nll':
        loss_fct = torch.nn.NLLLoss(reduction='none')
    elif loss_type == 'kl':
        loss_fct = torch.nn.KLDivLoss(log_target=True, reduction='none')

    loss = loss_fct(inputs, targets)  # 計算每個位置的 loss
    if loss_type == 'kl':
        loss = loss.sum(dim=1)        # KL 需要對 vocab 維度求和
    loss = loss[weights != 0]         # 只保留 weight != 0 的位置
    return loss.mean()                # 取平均
```

---

## 9. 參數更新機制

### 9.1 `loss.backward()` — 反向傳播

```python
loss.backward()
```

PyTorch 自動微分計算鏈：

```
loss
  ↑
loss = lm_loss + contrastive_loss + kl_loss
  ↑
lm_loss = CrossEntropy(logits, targets)
  ↑
logits = lm_head(hidden)
  ↑
hidden = Layer31(...Layer0(embedding, past_key_values=prefix)...)
  ↑
prefix = prefix_params[0], prefix_params[1], ...  ← 梯度在這裡累積
```

**梯度流向：**
- LLM 參數 → `requires_grad=False` → `grad = None`（不計算梯度）
- Prefix 參數 → `requires_grad=True` → `grad` 被計算並累積

### 9.2 `optimizer.step()` — AdamW 更新

```python
# 對每個可訓練的 prefix 參數
for param in prefix_params:
    grad = param.grad  # 累積的梯度

    # Adam 一階動量（梯度的移動平均）
    m = beta1 * m + (1 - beta1) * grad       # beta1=0.9

    # Adam 二階動量（梯度平方的移動平均）
    v = beta2 * v + (1 - beta2) * grad**2    # beta2=0.999

    # 偏差修正
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)

    # 計算更新量
    update = lr * m_hat / (sqrt(v_hat) + eps)  # lr=0.01, eps=1e-8

    # Weight Decay
    param.data -= lr * weight_decay * param.data  # weight_decay=0.01

    # 應用更新
    param.data -= update
```

### 9.3 數值範例

```python
# === 更新前 ===
prefix_params[0][0, 0, 0] = 0.0023
grad = 0.00062  # 累積了 2 步

# === Adam 計算（第 1 次更新, t=1）===
m = 0.9 * 0 + 0.1 * 0.00062 = 0.000062
v = 0.999 * 0 + 0.001 * 0.00062^2 = 3.844e-10

m_hat = 0.000062 / (1 - 0.9) = 0.00062
v_hat = 3.844e-10 / (1 - 0.999) = 3.844e-7

update = 0.01 * 0.00062 / (sqrt(3.844e-7) + 1e-8) ≈ 0.01

# === 更新後 ===
prefix_params[0][0, 0, 0] = 0.0023 - 0.01 = -0.0077
```

### 9.4 `scheduler.step()` — 學習率衰減

使用線性 warmup + 線性衰減：

```python
# warmup_steps = 0 → 沒有 warmup
# 線性衰減：lr = initial_lr * (1 - step / total_steps)
lr_step_100  = 0.01 * (1 - 100/25000) = 0.00996
lr_step_1000 = 0.01 * (1 - 1000/25000) = 0.0096
lr_step_25000 = 0.01 * (1 - 25000/25000) = 0.0
```

### 9.5 數值追蹤範例

```
時間點          | prefix_params[0][0,0,0] | grad     | 動作
初始化          | 0.0023                  | None     |
Step 0 backward | 0.0023                  | 0.00034  | 累積梯度
Step 1 backward | 0.0023                  | 0.00062  | 繼續累積
Step 1 update   | -0.0077                 | None     | ✅ 更新 + 清空
Step 2 backward | -0.0077                 | 0.00029  | 用新參數計算
Step 3 backward | -0.0077                 | 0.00051  | 繼續累積
Step 3 update   | -0.0128                 | None     | ✅ 更新 + 清空
...
Step 49999      | 0.1234                  | None     | 訓練完成
```

---

## 10. Prefix 與 LLM 的關係

### 10.1 核心關係

**Prefix 是 LLM 的「外掛控制器」，通過 attention 機制影響 LLM 的行為。**

```
預訓練 LLM (20億參數, 凍結)
       ↑
       │ 通過 attention 中的 Key/Value 拼接來影響
       │
Prefix 參數 (210萬參數, 可訓練)
```

### 10.2 Attention 機制的變化

**原始 Attention（無 Prefix）：**

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

**Prefix-Tuning Attention：**

$$\text{Attn}(Q, [K_{\text{prefix}}; K], [V_{\text{prefix}}; V]) = \text{softmax}\left(\frac{Q[K_{\text{prefix}}; K]^T}{\sqrt{d}}\right)[V_{\text{prefix}}; V]$$

### 10.3 梯度流向

```
Loss
  ↓ backward()
  │
  ├─→ LLM 參數 (transformer.*) ────→ grad = None（凍結）
  │
  └─→ Prefix 參數 (prefix_params.*) ─→ grad ≠ None → optimizer.step() → 更新
```

### 10.4 類比

| Fine-Tuning 全模型 | Prefix-Tuning |
|-------------------|---------------|
| 重新教育整個工程師 | 給工程師一份風格指南 |
| 改變所有知識 | 知識不變，只調整輸出方向 |
| 訓練 20 億參數 | 訓練 210 萬參數 (0.1%) |
| 保存 8 GB | 保存 2 MB |

---

## 11. 驗證與保存

### 11.1 每個 Epoch 結束的驗證

```python
if self.args.save_epochs > 0 and (idx+1) % self.args.save_epochs == 0:
    self.model.eval()
    with torch.no_grad():
        reported_eval_loss = self.do_eval()
    self.model.train()
```

`do_eval()` 的實現：

```python
def do_eval(self):
    val_sampler = SequentialSampler(self.val_dataset)  # 順序採樣
    val_dataloader = DataLoader(self.val_dataset, sampler=val_sampler, batch_size=1)
    acc_loss_dict = OrderedDict()
    for batch in val_dataloader:
        loss, loss_dict = self.step(batch)  # 同樣用 step()，但沒有 backward
        self.add_to_loss_dict(acc_loss_dict, loss_dict)
    return self.report_loss_dict(acc_loss_dict, len(val_dataloader))
```

### 11.2 保存 Checkpoint

每個 epoch 保存兩份：
1. `checkpoint-epoch-{n}` — 特定 epoch 的 checkpoint（永久保留）
2. `checkpoint-last` — 最新的 checkpoint（每次覆蓋）

```python
self.save(output_dir, global_step, idx+1, None, None)
self.save(last_output_dir, global_step, idx+1, None, None)
```

### 11.3 保存內容（`save_model()`）

```python
def save_model(model, path, args):
    # 只保存 prefix_params 的權重
    state_dict = model.prefix_params.state_dict()
    torch.save(state_dict, os.path.join(path, 'pytorch_model.bin'))  # ~2 MB

    # 保存模型配置
    model.config.save_pretrained(path)

    # 記錄原始預訓練模型路徑
    with open(os.path.join(path, 'lm.txt'), 'w') as f:
        f.write(args.pretrain_dir)  # e.g. 'Salesforce/codegen-2B-multi'
```

**不保存** LLM 權重（因為從未修改）。

### 11.4 Checkpoint 目錄結構

```
trained/2b-prefix/
├── checkpoint-epoch-1/
│   ├── pytorch_model.bin     ← prefix_params 權重 (~2 MB)
│   ├── config.json           ← 模型配置
│   ├── lm.txt                ← 預訓練模型路徑
│   ├── tokenizer_config.json
│   ├── step_file.txt         ← 當前 global_step
│   └── epoch_file.txt        ← 當前 epoch
├── checkpoint-epoch-2/
├── ...
├── checkpoint-epoch-5/
└── checkpoint-last/
```

### 11.5 載入 Checkpoint（推理時）

```python
def load_model(model_type='prefix', path, is_training=False, args):
    # 從 lm.txt 讀回預訓練模型路徑
    with open(os.path.join(path, 'lm.txt')) as f:
        lm_path = f.read()  # 'Salesforce/codegen-2B-multi'

    # 重新載入 LLM + 建立 prefix 結構
    model = model_from_pretrained(lm_path, 'prefix', lm_config)

    # 載入訓練好的 prefix 參數
    prefix_file = os.path.join(path, 'pytorch_model.bin')
    model.prefix_params.load_state_dict(torch.load(prefix_file))
```

---

## 12. 超參數配置

### 12.1 模型相關

| 參數 | 350M | 2B | 6B | 說明 |
|------|:----:|:--:|:--:|------|
| `n_prefix_token` | 5 | 8 | 12 | 每層的 prefix token 數 |
| `num_train_epochs` | 8 | 5 | 5 | 訓練 epoch 數 |
| `kl_loss_ratio` | 1600 | 1600 | 2000 | KL loss 係數（÷1000） |
| `learning_rate` | 1e-2 | 1e-2 | 1e-2 | 初始學習率 |

### 12.2 訓練相關

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `batch_size` | 1 | 每個 batch 的樣本數 |
| `grad_acc_steps` | 2 | 梯度累積步數 |
| `max_num_tokens` | 1024 | 最大 token 長度 |
| `max_grad_norm` | 1.0 | 梯度裁剪閾值 |
| `weight_decay` | 0.01 | L2 正則化係數 |
| `adam_epsilon` | 1e-8 | Adam 穩定性常數 |
| `warmup_steps` | 0 | 學習率 warmup 步數 |
| `dropout` | 0.1 | Prefix dropout 率 |
| `diff_level` | mix | 差異標記層級 |

### 12.3 Loss 相關

| 參數 | 預設值 | 實際權重 | 說明 |
|------|--------|---------|------|
| `lm_loss_ratio` | 1 | 1 | LM loss 係數 |
| `contrastive_loss_ratio` | 400 | 4.0 (÷100) | 對比 loss 係數 |
| `kl_loss_ratio` | 1600 | 1.6 (÷1000) | KL loss 係數 |

### 12.4 記錄與保存

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `logging_steps` | 100 | 每 N 次更新記錄一次日誌 |
| `save_epochs` | 1 | 每 N 個 epoch 保存一次 |
| `seed` | 1 | 隨機種子 |

---

## 13. 訓練 vs 推理

### 13.1 對比

| 特性 | 訓練時 | 推理/生成時 |
|------|--------|-----------|
| **輸入** | 完整程式碼 | 只有 prompt |
| **輸出** | Loss 標量 | 生成的程式碼 |
| **Forward 次數** | 3 次/batch（正確、錯誤、無 prefix） | 每生成 1 個 token 做 1 次 |
| **Prefix 參數** | 持續更新 | 固定不變 |
| **模型模式** | `model.train()`（啟用 dropout） | `model.eval()`（停用 dropout） |
| **是否生成** | 不生成 | 逐 token 生成 |

### 13.2 訓練時的 Forward（Teacher Forcing）

```python
# 一次性計算所有位置的 logits
inputs = [def, query, (, name, ), :, \n, cursor, ., execute, ...]
                                                                ↑ 已知所有 token
outputs = model(inputs, past_key_values=prefix)
logits = outputs.logits  # (1, 512, 51200)

# 不生成，只計算 loss
loss = CrossEntropy(logits[:, :-1], inputs[:, 1:])
```

### 13.3 推理時的 Forward（Autoregressive Generation）

```python
prompt = [def, query, (, name, ), :]
generated = prompt.copy()

for _ in range(max_new_tokens):
    outputs = model(generated, past_key_values=prefix)
    next_token = argmax(outputs.logits[:, -1, :])
    generated.append(next_token)
    if next_token == EOS: break
```

---

## 14. 完整流程總覽

```
啟動: train.py → PrefixTrainer(args).run()
        │
        ▼
[1] load_model()
    載入 CodeGen-2B (20億參數，凍結)
    建立 128 個 prefix 參數張量 (210萬參數，可訓練)
        │
        ▼
[2] load_dataset()
    讀取 cwe-089.jsonl, cwe-125.jsonl, ... 等安全漏洞 diff 資料
    每筆 JSON → 2 筆訓練資料 (secure + vulnerable)
    每筆 = (tokens, weights, control_id, vul_id)
        │
        ▼
[3] 訓練迴圈 (5 epochs × N samples)
    ┌──────────────────────────────────────────────────────────────┐
    │ 取一筆 batch:                                                │
    │   tokens = [318, 12405, 7, ...]                             │
    │   weights = [0, 0, 0, ..., 1, 1, 1, ..., 0, 0]             │
    │   control_id = 0 (secure)                                    │
    │                                                              │
    │ Forward 1: model(tokens, prefix=secure)     → correct_logits │
    │ Forward 2: model(tokens, prefix=vulnerable) → incorrect_logits│
    │ Forward 3: model(tokens, prefix=None)       → ref_logits     │
    │                                                              │
    │ lm_loss:          差異 token 上的 cross-entropy              │
    │ contrastive_loss: 差異 token 上正確 vs 錯誤 prefix 的對比    │
    │ kl_loss:          非差異 token 上與原始 LLM 的 KL 散度       │
    │                                                              │
    │ loss = lm_loss + contrastive_loss + kl_loss                  │
    │ loss.backward() → 只更新 prefix_params                       │
    └──────────────────────────────────────────────────────────────┘
        │
        ▼
[4] 每個 Epoch 結束
    ├─ model.eval() → 驗證集評估
    ├─ model.train() → 切回訓練模式
    └─ save() → 保存 prefix 參數 (~2 MB) + config + lm.txt
        │
        ▼
[5] 訓練完成
    trained/2b-prefix/checkpoint-last/ 包含最終的 prefix 參數
    可搭配原始 LLM 進行安全/漏洞程式碼生成
```
