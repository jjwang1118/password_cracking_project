# Prefix Model 架構說明

## 概述

Prefix-tuning 是一種參數高效的微調方法，通過在每層 Transformer 的 attention 機制中注入可學習的虛擬 key-value 對，實現對模型行為的控制，而無需修改原始 LLM 的參數。

本項目實現了三種 Prefix 模型：
- `CodeGenPrefixCausalLM` (基於 Salesforce CodeGen)
- `IncoderPrefixLM` (基於 Facebook Incoder)
- `SantaPrefixLM` (基於 BigCode SantaCoder)

---

## CodeGenPrefixCausalLM 類結構

### 繼承關係

```python
class CodeGenPrefixCausalLM(CodeGenForCausalLM):
```

繼承自 HuggingFace 的 `CodeGenForCausalLM`，擴展了 prefix-tuning 功能。

---

## 核心組件

### 1. Prefix 參數初始化

#### 代碼

```python
def __init__(self, config):
    super().__init__(config)
    
    self.n_embed_per_head = config.n_embd // config.n_head
    self.prefix_params = torch.nn.ParameterList()
    for _ in range(config.n_control):      # 2 種控制信號 (安全/不安全)
        for _ in range(config.n_layer):    # 每一層
            for _ in range(2):             # key 和 value
                param_size = (config.n_head, config.n_prefix_token, self.n_embed_per_head)
                param = torch.nn.Parameter(torch.zeros(param_size, requires_grad=True))
                self.prefix_params.append(param)
    self.dropout = torch.nn.Dropout(config.prefix_dropout)
```

#### 參數結構

**總參數量**：`n_control × n_layer × 2 × (n_head × n_prefix_token × n_embed_per_head)`

實際數量：`2 × n_layer × 2 × hidden_size × n_prefix_token`

**示例**（CodeGen-350M，n_layer=20, hidden_size=1024, n_prefix_token=10）：
```
4 × 20 × 1024 × 10 = 819,200 參數 (約 0.8M)
```

#### 三層迴圈結構

| 迴圈層級 | 範圍 | 含義 |
|---------|------|------|
| 外層 | `n_control = 2` | 控制信號數量（安全/不安全） |
| 中層 | `n_layer` | Transformer 層數 |
| 內層 | `2` | 每層的 key 和 value |

#### 參數維度說明

```python
param_size = (n_head, n_prefix_token, n_embed_per_head)
```

- **`n_head`**：Multi-head attention 的頭數（如 16）
  - 每個 head 獨立學習不同的注意力模式
  
- **`n_prefix_token`**：虛擬 prefix 的長度（超參數，如 5-20）
  - 假裝在序列前有幾個虛擬 token
  
- **`n_embed_per_head`**：每個 head 的維度 = `hidden_size / n_head`
  - 將總維度平均分配給每個 head

#### 線性存儲結構

```
索引映射（n_layer=3 示例）：
[0]  control=0, layer=0, key
[1]  control=0, layer=0, val
[2]  control=0, layer=1, key
[3]  control=0, layer=1, val
[4]  control=0, layer=2, key
[5]  control=0, layer=2, val
[6]  control=1, layer=0, key
[7]  control=1, layer=0, val
[8]  control=1, layer=1, key
[9]  control=1, layer=1, val
[10] control=1, layer=2, key
[11] control=1, layer=2, val
```

**索引計算公式**：
```python
key_idx = control_id * n_layer * 2 + layer_id * 2
val_idx = key_idx + 1
```

#### 為何初始化為零？

```python
torch.zeros(param_size, requires_grad=True)
```

零初始化讓模型一開始行為與原始 LLM 相同，訓練過程中逐漸學習差異化的控制能力。

---

### 2. Prefix 轉換為 Past Key Values

#### 代碼

```python
def get_past_from_prefix(self, control_ids):
    past = list()
    for i in range(self.config.n_layer):
        past.append(list())
        key_stack, val_stack = [], []
        for control_id in control_ids:
            key_idx = control_id * self.config.n_layer * 2 + i * 2
            val_idx = key_idx + 1
            key = self.dropout(self.prefix_params[key_idx])
            val = self.dropout(self.prefix_params[val_idx])
            key_stack.append(key)
            val_stack.append(val)
        past[i].append(torch.stack(key_stack))
        past[i].append(torch.stack(val_stack))
    return past
```

#### 功能說明

將訓練好的 prefix 參數轉換為 Transformer 可用的 `past_key_values` 格式。

#### 執行流程

1. **初始化結構**：為每一層創建空列表
2. **遍歷層**：處理每個 Transformer 層
3. **處理 batch**：為 batch 中每個樣本取對應的 control_id
4. **索引計算**：定位到 prefix_params 中的對應參數
5. **應用 Dropout**：訓練時隨機丟棄部分值（正則化）
6. **堆疊 batch**：用 `torch.stack` 合併為 batch 格式

#### 輸出結構

```python
past_key_values = [
    # Layer 0
    [key_tensor, value_tensor],  # shape: [batch, n_head, n_prefix_token, embed_per_head]
    # Layer 1
    [key_tensor, value_tensor],
    ...
    # Layer N-1
    [key_tensor, value_tensor]
]
```

**維度**：`[n_layer, 2]`
- 第一維：層數
- 第二維：key 和 value

#### 示例

```python
# 假設 batch_size=2, control_ids=[0, 1]
past = get_past_from_prefix([0, 1])

# past[0][0] = key tensor for layer 0
# shape: [2, 16, 10, 64]
#         ↑   ↑   ↑   ↑
#      batch head prefix embed_per_head
```

---

### 3. 代生成輸入準備

#### 代碼

```python
def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
    token_type_ids = kwargs.get("token_type_ids", None)
    if past:
        # 後續生成步驟：只保留最後一個 token
        input_ids = input_ids[:, -1].unsqueeze(-1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
    else:
        # 首次生成：創建 prefix
        control_ids = [kwargs['control_id']] * input_ids.shape[0]
        past = self.get_past_from_prefix(control_ids)

    return {
        "input_ids": input_ids,
        "past_key_values": past,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": None,
        "attention_mask": None,
        "token_type_ids": token_type_ids,
    }
```

#### 功能說明

在文本生成過程中被 HuggingFace 自動調用，用於準備每一步的輸入。

#### 兩種情況

**情況 1：首次生成 (`past=None`)**

```python
control_ids = [kwargs['control_id']] * input_ids.shape[0]
past = self.get_past_from_prefix(control_ids)
```

- 發生時機：生成第一個 token
- 從 `kwargs` 取得 `control_id`（0 或 1）
- 複製 `batch_size` 次：`[0] * 2 = [0, 0]`
- 生成 prefix 的 KV cache

**情況 2：後續生成 (`past` 已存在)**

```python
input_ids = input_ids[:, -1].unsqueeze(-1)
```

- 發生時機：生成第 2、3、4... 個 token
- 只保留最後一個 token
- 前面的 token 已在 `past_key_values` 中

**為何這樣做？**

```
第 1 步：input_ids = [10, 20, 30]         → past = None
第 2 步：input_ids = [10, 20, 30, 40]     → 只需 [40]
第 3 步：input_ids = [10, 20, 30, 40, 50] → 只需 [50]
```

#### 生成流程

```
用戶調用: model.generate(..., control_id=0)
    ↓
Step 1: prepare_inputs_for_generation()
  - past = None
  - 創建 prefix past_key_values
  - input_ids = [提示詞的所有 token]
    ↓
Step 2: forward() → 輸出第 1 個新 token
    ↓
Step 3: prepare_inputs_for_generation()
  - past 已存在
  - input_ids = [剛生成的 1 個 token]
    ↓
Step 4: forward() → 輸出第 2 個新 token
    ↓
重複直到生成結束...
```

---

### 4. Forward 傳播

#### 代碼

```python
def forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    control_id = None,  # placeholder，實際未使用
) -> Union[Tuple, CausalLMOutputWithPast]:
    return super().forward(
        input_ids=input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict
    )
```

#### 功能說明

直接調用父類的 forward 方法，prefix 通過 `past_key_values` 參數傳入。

**`control_id` 參數**：僅用於通過 HuggingFace 的參數檢查，實際在 `prepare_inputs_for_generation()` 中使用。

---

## Control ID 定義

在 `sven/constant.py` 中定義：

```python
SEC_LABEL = 'sec'   # control_id = 0 (安全代碼)
VUL_LABEL = 'vul'   # control_id = 1 (不安全代碼)
BINARY_LABELS = [SEC_LABEL, VUL_LABEL]
```

---

## 訓練機制

### 參數凍結

在 `sven/trainer.py` 中：

```python
for n, p in self.model.named_parameters():
    if n.startswith('prefix_params'):
        p.requires_grad = True   # 只有 prefix 可訓練
    else:
        p.requires_grad = False  # 凍結原始 LLM 參數
```

### 訓練損失

1. **LM Loss**：使用正確的 control_id 計算語言模型損失
2. **Contrastive Loss**（可選）：對比正確/錯誤 control 的預測
3. **KL Loss**（可選）：讓非目標 token 保持與原始模型相似的分布

```python
# 正確控制信號的 logits
correct_logits = get_logits_from_lm(model, inputs, control_ids)

# 錯誤控制信號（翻轉：0→1, 1→0）
incorrect_control_ids = -1 * (control_ids - 1)
incorrect_logits = get_logits_from_lm(model, inputs, incorrect_control_ids)
```

---

## 模型配置參數

### CodeGen 模型

```python
n_layer      # Transformer 層數
n_head       # Attention 頭數
n_embd       # 隱藏層維度
n_prefix_token    # Prefix 長度（超參數）
prefix_dropout    # Dropout 率
n_control    # 控制信號數量（固定為 2）
```

### Incoder 模型

```python
num_layers        # Transformer 層數
attention_heads   # Attention 頭數
d_model          # 隱藏層維度
```

### SantaCoder 模型

使用與 CodeGen 相同的參數名。

---

## Dropout 機制

```python
self.dropout = torch.nn.Dropout(config.prefix_dropout)
```

### 作用

在訓練時隨機將 prefix 參數中的部分值設為 0，用於正則化。

### 使用位置

```python
key = self.dropout(self.prefix_params[key_idx])
val = self.dropout(self.prefix_params[val_idx])
```

### 訓練 vs 推理

- **訓練模式** (`model.train()`)：以 `prefix_dropout` 機率隨機丟棄
- **推理模式** (`model.eval()`)：Dropout 自動關閉

---

## Prefix 機制原理

### Attention 中的作用

```
標準 Attention:
Q @ K^T → softmax → @ V

加入 Prefix 後:
Q @ [prefix_K ; real_K]^T → softmax → @ [prefix_V ; real_V]
     └─虛擬 token─┘ └真實 token┘
```

### 視覺化

```
原始輸入序列:  [def] [foo] [(] [x] [)] [:]

實際 attention: [P0] [P1] [P2] [P3] [P4] [def] [foo] [(] [x] [)] [:]
                └──────虛擬 prefix──────┘ └────真實 token────────┘
```

這些虛擬 token **不是真實輸入**，而是學習到的 key-value 向量，用來引導模型的注意力。

---

## 格式規定

### Past Key Values 格式

這個格式由 **HuggingFace Transformer 的 `past_key_values` 介面**規定：

```python
past_key_values = (
    # Layer 0
    (key, value),  # shape: (batch, n_head, seq_len, embed_per_head)
    # Layer 1
    (key, value),
    ...
)
```

### 為何必須這樣設計？

```
標準 Attention 計算:
Q @ K^T → (batch, n_head, seq_len, seq_len)
        → softmax
        → @ V → (batch, n_head, seq_len, embed_per_head)

Prefix 注入後:
Q @ [prefix_K ; real_K]^T 
  → prefix_K 必須是 (batch, n_head, n_prefix_token, embed_per_head)
     才能和 real_K concat 在一起
```

所以 `n_head` 和 `n_embed_per_head` 是被模型架構強制規定的，只有 `n_prefix_token` 是可調的超參數。

---

## 數據流程

```
訓練數據 (安全/不安全代碼對)
         ↓
   PrefixDataset 處理
   - control_id: 0(安全) / 1(不安全)
   - weights: 標記需計算 loss 的 token
         ↓
   Prefix 參數 → get_past_from_prefix() → past_key_values
         ↓
   LLM Forward (原始參數凍結)
         ↓
   Loss 計算 + 反向傳播 → 僅更新 prefix_params
```

---

## 總結

Prefix 本質上是**虛擬的 key-value 向量**，預先注入到每層 transformer 的 attention 機制中。透過訓練，不同的 control_id 對應的 prefix 學會引導模型生成安全或不安全的代碼，而無需修改原始 LLM 的參數。

這種方法：
- **參數高效**：只訓練約 0.8M 參數 vs 原始模型的數百 M 或數 B 參數
- **靈活切換**：推理時可自由選擇 control_id
- **保持原模型**：不破壞預訓練知識
