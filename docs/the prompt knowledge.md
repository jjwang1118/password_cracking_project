# PassLLM 訓練資料格式說明

本文檔詳細說明 PassLLM 框架中 Trawling Attack 和 Targeted Attack 的訓練資料格式、設計原理以及模型訓練機制。

---

## 目錄
1. [Trawling Attack 資料格式](#trawling-attack-資料格式)
2. [Targeted Attack 資料格式](#targeted-attack-資料格式)
3. [Loss 計算機制](#loss-計算機制)
4. [Prompt 設計原理](#prompt-設計原理)
5. [模型學習機制](#模型學習機制)
6. [自定義訓練資料](#自定義訓練資料)

---

## Trawling Attack 資料格式

### 訓練階段格式

```
[BOS] + Prompt + Password + [EOS]
```

**完整範例：**
```
<BOS>As a trawling password guessing model, your task is to generate user's passwords.
Password:123456<EOS>
```

**組成部分：**
- **BOS Token**: 開始標記（如果模型有的話）
- **Prompt**: 固定的指令文本
- **Password**: 目標密碼字串
- **EOS Token**: 結束標記

### 推理階段格式

只提供 Prompt：
```
<BOS>As a trawling password guessing model, your task is to generate user's passwords.
Password:
```

模型從 `Password:` 後開始生成密碼。

### 資料來源示例

[data/rockyou/rockyou_0.02_test.txt](../data/rockyou/rockyou_0.02_test.txt)

```
ronda
ishockey
Aries1953
kathleen20
AMORS
...
```

每行一個密碼，獨立處理。

### 代碼實現

**訓練資料處理：** [src/utils/tokenize.py](../src/utils/tokenize.py#L82-L104)

```python
def process_train_trawling(example, tokenizer, prompt_id, vocab:dict, MAX_LENGTH=256, IGNORE_INDEX=-100):
    # 1. 生成 Prompt
    indice = generate_prompt_trawling(prompt_id)
    if tokenizer.bos_token:
        indice = tokenizer.bos_token + indice
    
    # 2. 編碼 Prompt 和 Password
    response = example["password"]
    encoded_indice = tokenizer(indice, add_special_tokens=False)
    encoded_response = encode_limit(response, vocab)
    
    # 3. 組合序列
    input_ids = encoded_indice["input_ids"] + encoded_response["input_ids"] + [tokenizer.eos_token_id]
    
    # 4. 設定 labels（只計算 Password 的 loss）
    labels = input_ids.copy()
    labels[:len(encoded_indice["input_ids"])] = [IGNORE_INDEX] * len(encoded_indice["input_ids"])
    
    return {"input_ids": input_ids, "attention_mask": ..., "labels": labels}
```

---

## Targeted Attack 資料格式

### 訓練階段格式

```
[BOS] + Prompt + Knowledge_JSON + Password + [EOS]
```

**完整範例：**
```
<BOS>As a targeted password guessing model, your task is to utilize the provided account information to guess the password.{"Old password": "123456"}chenxp1997<EOS>
```

**組成部分：**
- **BOS Token**: 開始標記
- **Prompt**: 固定的指令文本
- **Knowledge JSON**: 用戶背景信息（如舊密碼、用戶名等）
- **Password**: 目標密碼
- **EOS Token**: 結束標記

### 推理階段格式

只提供 Prompt + Knowledge：
```
<BOS>As a targeted password guessing model, your task is to utilize the provided account information to guess the password.{"Old password": "123456"}
```

模型從 Knowledge JSON 後開始生成密碼。

### 資料來源示例

[data/126_csdn/126_csdn_test.json](../data/126_csdn/126_csdn_test.json)

```json
[
  {
    "Knowledge": {
      "Old password": "shilpa"
    },
    "password": "9702910513"
  },
  {
    "Knowledge": {
      "Old password": "123456"
    },
    "password": "chenxp1997"
  },
  {
    "Knowledge": {
      "Old password": "foolish"
    },
    "password": "foolish516"
  }
]
```

### 密碼關聯模式

從數據集可觀察到不同的密碼變化模式：

#### ✅ 有明顯關聯
```
Old: "foolish"    → New: "foolish516"      (加數字後綴)
Old: "1986726"    → New: "BO1986726"       (加前綴)
Old: "720310"     → New: "7203103518"      (延伸數字)
Old: "5678552"    → New: "56785525678552"  (重複密碼)
Old: "lengmian"   → New: "lengmian412827"  (加數字後綴)
```

#### ❌ 看似無關
```
Old: "shilpa"     → New: "9702910513"      (可能是電話號碼)
Old: "839953"     → New: "jiangdong"       (完全不同)
Old: "woaixue"    → New: "yanglijuan"      (完全不同)
```

### 代碼實現

**訓練資料處理：** [src/utils/tokenize.py](../src/utils/tokenize.py#L108-L137)

```python
def process_train_targeted(example, tokenizer, vocab, prompt_id, 
                          mask_indice=True, mask_pii=True, 
                          MAX_LENGTH=512, IGNORE_INDEX=-100):
    # 1. 生成 Prompt 和 Knowledge
    indice, content, _ = generate_prompt_targeted(example, prompt_id)
    if tokenizer.bos_token:
        indice = tokenizer.bos_token + indice
    
    # 2. 編碼各部分
    response = example["password"]
    encoded_indice = tokenizer(indice, add_special_tokens=False)
    encoded_content = tokenizer(content, add_special_tokens=False)
    encoded_response = encode_limit(response, vocab)
    
    # 3. 組合序列
    input_ids = encoded_indice["input_ids"] + encoded_content["input_ids"] + \
                encoded_response["input_ids"] + [tokenizer.eos_token_id]
    
    # 4. 設定 labels（Mask Prompt 和 Knowledge）
    labels = input_ids.copy()
    if mask_indice:
        labels[:len(encoded_indice["input_ids"])] = [IGNORE_INDEX] * len(encoded_indice["input_ids"])
    if mask_pii:
        start = len(encoded_indice["input_ids"])
        end = start + len(encoded_content["input_ids"])
        labels[start:end] = [IGNORE_INDEX] * len(encoded_content["input_ids"])
    
    return {"input_ids": input_ids, "attention_mask": ..., "labels": labels}
```

---

## Loss 計算機制

### Trawling Attack

```
完整輸入序列：
┌─────────────┬──────────────┬─────┐
│   Prompt    │   Password   │ EOS │
└─────────────┴──────────────┴─────┘

Labels (計算 Loss)：
┌─────────────┬──────────────┬─────┐
│  IGNORE     │  ✅ 計算 Loss │ ✅  │
│  (-100)     │              │     │
└─────────────┴──────────────┴─────┘
```

### Targeted Attack

```
完整輸入序列：
┌─────────────┬──────────────────┬──────────────┬─────┐
│   Prompt    │   Knowledge      │   Password   │ EOS │
└─────────────┴──────────────────┴──────────────┴─────┘

Labels (計算 Loss)：
┌─────────────┬──────────────────┬──────────────┬─────┐
│  IGNORE     │    IGNORE        │  ✅ 計算 Loss │ ✅  │
│  (-100)     │    (-100)        │              │     │
│ mask_indice │   mask_pii       │              │     │
└─────────────┴──────────────────┴──────────────┴─────┘
```

### Loss 計算原理

在 PyTorch 中使用 `CrossEntropyLoss(ignore_index=-100)`：

```python
# 只有 label != -100 的位置會計算 loss
loss_fn = CrossEntropyLoss(ignore_index=-100)

# Prompt 和 Knowledge 被設為 -100，不參與 loss 計算
# 只有 Password 和 EOS 的預測錯誤會產生 loss 和梯度
```

### 為什麼要 Mask？

| 部分 | 是否計算 Loss | 原因 |
|------|--------------|------|
| **Prompt** | ❌ 不計算 | 固定指令，不需要學習生成 |
| **Knowledge** | ❌ 不計算 | 已知信息，不需要學習重複 |
| **Password** | ✅ 計算 | **核心目標**：學習生成正確密碼 |
| **EOS** | ✅ 計算 | 學習何時停止生成 |

**好處：**
1. **專注目標任務**：強迫模型學習 Knowledge → Password 的映射
2. **節省計算**：不浪費資源在重複已知信息
3. **訓練高效**：Loss 信號直接針對核心目標

---

## Prompt 設計原理

### Trawling Prompt：為什麼有 `\nPassword:`？

**Prompt 內容：**
```python
"As a trawling password guessing model, your task is to generate user's passwords.\nPassword:"
```

**實際顯示：**
```
As a trawling password guessing model, your task is to generate user's passwords.
Password:█  ← 模型從這裡開始生成
```

**設計原因：**

1. **明確生成起點**
   - 沒有中間的 Knowledge 作為分界
   - 需要明確標記告訴模型"從這裡開始生成密碼"

2. **避免模型生成多餘內容**
   ```
   不好的設計（沒有 Password:）：
   Input: "...generate user's passwords."
   Output: "Sure, here are some passwords: 123456, password..."
   ❌ 格式不統一，難以解析
   
   好的設計（有 Password:）：
   Input: "...generate user's passwords.\nPassword:"
   Output: "123456"
   ✅ 直接生成密碼
   ```

3. **Prompt Engineering 最佳實踐**
   ```python
   # 給模型提供結構化模板
   指令 + "\n" + 標籤 + ":" → 讓模型填空
   
   類似：
   "Translate to French.\nFrench:"
   "Summarize this text.\nSummary:"
   "Generate password.\nPassword:"
   ```

### Targeted Prompt：為什麼沒有 `Password:`？

**Prompt 內容：**
```python
"As a targeted password guessing model, your task is to utilize the provided account information to guess the password."
```

**完整格式：**
```
As a targeted password guessing model, your task is to utilize the provided account information to guess the password.{"Old password": "123456"}chenxp1997
```

**設計原因：**

1. **JSON 提供天然分界**
   ```
   ...guess the password.{"Old password": "123456"}chenxp1997
                         ↑                        ↑
                      JSON 開始              JSON 結束，密碼開始
   ```
   - `{...}` 結構已經是明確的邊界標記
   - 模型看到 `}` 後就知道該生成密碼了

2. **避免語義混淆**
   ```
   如果加 Password: 在 Knowledge 前：
   ...password.Password:{"Old password": "123456"}chenxp1997
   ❌ "Password:" 後面接的是 Knowledge，不是密碼，語義混淆
   
   如果加 Password: 在 Knowledge 後：
   ...password.{"Old password": "123456"}Password:chenxp1997
   ✅ 可行但不必要，JSON 已經足夠
   ```

3. **保持簡潔**
   - 不添加不必要的標記
   - 實驗證明這種格式模型可以正確理解

### 設計對比

| 特性 | Trawling | Targeted |
|------|----------|----------|
| **結構** | Prompt + Password | Prompt + Knowledge + Password |
| **分界標記** | `\nPassword:` | Knowledge JSON 本身 |
| **為何如此** | 沒有中間內容，需要明確標記 | JSON 提供天然邊界 |
| **完整示例** | `...passwords.\nPassword:123456` | `...password.{"Old password": "123456"}chenxp1997` |

---

## 模型學習機制

### 常見疑問：Knowledge 被 Mask 了，模型如何學習關聯？

雖然 Knowledge 被 mask（不計算 loss），但模型**完全可以學到 Knowledge 和 Password 的關聯**。

### Transformer Self-Attention 機制

```
預測密碼時的 Attention：

┌─────────┬───────────┬─────┬─────┬─────┬─────┐
│ Prompt  │ Knowledge │  c  │  h  │  e  │  n  │
└─────────┴───────────┴─────┴─────┴─────┴─────┘
                ↓         ↓
                └─────────┴──→ 預測下一個字元 'x'
```

當模型預測密碼的每個字元時：
- **會 attend 到前面所有 tokens**，包括 Knowledge
- Self-attention 會計算 Knowledge 和當前位置的關聯性
- 模型**必須利用 Knowledge 信息**來做出正確預測

### Forward Pass vs Loss Calculation

```python
# Forward Pass (前向傳播) - 模型看到所有內容
input: [Prompt] + [Knowledge] + [Password]
       ↓         ↓              ↓
模型處理:  ✅        ✅             ✅  (全部都會處理)

# Loss Calculation (只計算 Password 的錯誤)
labels: [IGNORE] + [IGNORE] + [Password]
             ❌        ❌          ✅  (只這裡算 loss)
```

**關鍵理解：**
- **IGNORE 不是"模型看不到"**
- **IGNORE 只是"這部分預測錯了也不罰分"**
- 模型在預測 Password 時，仍然完整看到並使用 Knowledge

### 梯度回傳路徑

```
Loss 計算流程：
┌─────────────────────────────────────────┐
│ 預測密碼字元時用到了 Knowledge 的信息    │
│         ↓                               │
│ 如果預測錯誤 → 產生 Loss                │
│         ↓                               │
│ 梯度回傳 → 調整模型參數                 │
│         ↓                               │
│ 包括：如何理解 Knowledge 的權重         │
│       如何將 Knowledge 映射到 Password  │
└─────────────────────────────────────────┘
```

**具體範例：**

訓練樣本：
```
Knowledge: {"Old password": "123456"}
Password: chenxp1997
```

- 模型看到 `"123456"`，預測應該生成 `"chenxp1997"`
- 如果預測錯誤（比如生成 `"abcd1234"`），Loss 增加
- 梯度回傳會調整：
  - Attention 權重 → 讓模型更關注 Knowledge
  - 特徵提取 → 學習 `"123456"` 和 `"chenxp1997"` 的關係

### 類比理解

**就像考試：**
- 題目（Knowledge）不算分
- 答案（Password）算分
- 但你**必須看懂題目**才能答對
- 所以大腦仍然會學習如何理解題目和答案的關係

---

## 自定義訓練資料

### 場景：只有 {帳號、密碼} 的資料

如果你只有帳號和密碼，可以按以下方式設計：

#### 方案 1：帳號作為 Knowledge（推薦）

**資料格式：**
```json
[
  {
    "Knowledge": {
      "Username": "john_doe"
    },
    "password": "johndoe123"
  },
  {
    "Knowledge": {
      "Username": "alice2024"
    },
    "password": "alice@2024"
  }
]
```

**訓練格式：**
```
<BOS>As a targeted password guessing model, your task is to utilize the provided account information to guess the password.{"Username": "john_doe"}johndoe123<EOS>
```

**模型會學到：**
- 用戶名和密碼的相似性（如 `john_doe` → `johndoe123`）
- 常見變體（加數字、特殊符號、大小寫變化）
- 拼音/英文名與密碼的關聯

#### 方案 2：從帳號提取特徵（進階）

如果帳號包含多種信息（如 email），可以拆分：

```json
[
  {
    "Knowledge": {
      "Username": "john.doe",
      "Email_prefix": "john.doe",
      "Email_domain": "gmail.com"
    },
    "password": "johndoe@gmail"
  }
]
```

#### 方案 3：數據增強

從帳號中提取特徵：

```json
[
  {
    "Knowledge": {
      "Username": "alice2024",
      "Name_part": "alice",
      "Number_part": "2024"
    },
    "password": "Alice@2024"
  }
]
```

### 實現步驟

#### Step 1: 準備數據

創建 `prepare_data.py`：

```python
import json
import pandas as pd

def prepare_training_data(input_csv, output_json):
    """
    將 CSV 格式 (username, password) 轉換成訓練格式
    """
    df = pd.read_csv(input_csv)
    
    data = []
    for idx, row in df.iterrows():
        data.append({
            "Knowledge": {
                "Username": row["username"]
            },
            "password": row["password"]
        })
    
    # 分割訓練集和驗證集 (95% / 5%)
    split_idx = int(len(data) * 0.95)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # 保存
    with open(f"{output_json}_train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(f"{output_json}_val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 訓練集: {len(train_data)} 樣本")
    print(f"✅ 驗證集: {len(val_data)} 樣本")

# 使用範例
if __name__ == "__main__":
    prepare_training_data(
        input_csv="your_accounts.csv",  # 包含 username,password 欄位
        output_json="data/my_dataset/data"
    )
```

#### Step 2: 創建配置文件

創建 `config/my_training_config.ini`：

```ini
[global]
do_train = true
do_eval = false
seo = targeted

[train.basic]
model = model/Qwen2.5-0.5B-Instruct
tokenizer = model/Qwen2.5-0.5B-Instruct
train_data = data/my_dataset/data_train.json
validation_data = data/my_dataset/data_val.json

[train.trainer]
prompmt_template_id = 0
precistion = half

[train.lora]
r = 16
target_modules = ['q_proj','k_proj','v_proj']
lora_alpha = 32
lora_dropout = 0.2
bias = none 
init_lora_weights = true 

[train.hyperparameters]
output_dir = checkpoints/my_model/
per_device_train_batch_size = 4
gradient_accumulation_steps = 16
weight_decay = 0.01
learning_rate = 5e-4
num_train_epochs = 3
warmup_ratio = 0.1
logging_steps = 10
save_steps = 200
```

#### Step 3: 開始訓練

```powershell
# 1. 準備數據
python prepare_data.py

# 2. 開始訓練
python main.py --mode train --config config/my_training_config.ini
```

### 關鍵考量

#### 1. 數據質量比數量重要
- ✅ 帳號和密碼有關聯的樣本（如 `john` → `john123`）
- ❌ 完全隨機無關的樣本（如 `user123` → `xK9$mP2q`）

#### 2. 最少數據量建議
- **訓練集**: 至少 10,000 組
- **驗證集**: 1,000-2,000 組
- 如果數據太少，模型難以學到有效模式

#### 3. 帳號特徵越豐富越好
```json
// 好的範例（信息豐富）
{"Username": "john_doe_1990"}
{"Username": "alice.wang@company.com"}

// 差的範例（信息太少）
{"Username": "user123"}
{"Username": "a"}
```

---

## 總結

### Trawling vs Targeted

| 特性 | Trawling | Targeted |
|------|----------|----------|
| **訓練格式** | `[BOS] + Prompt + Password + [EOS]` | `[BOS] + Prompt + Knowledge + Password + [EOS]` |
| **用戶信息** | ❌ 無 | ✅ 有 (Knowledge JSON) |
| **Prompt 標籤** | `\nPassword:` | 無（JSON 提供分界） |
| **Loss 計算** | 只在 Password | 只在 Password (Knowledge 被 mask) |
| **應用場景** | 通用密碼破解 | 針對特定用戶（有背景信息） |
| **數據示例** | `rockyou.txt` | `126_csdn.json` |

### 核心原則

1. **每個密碼獨立處理** - 不會將多個密碼串接
2. **只計算目標部分的 Loss** - Prompt 和 Knowledge 被 mask
3. **模型能學到關聯** - 雖然 Knowledge 不算分，但通過 attention 機制仍會學習
4. **Prompt 設計因任務而異** - 有中間結構就不需要額外標籤

