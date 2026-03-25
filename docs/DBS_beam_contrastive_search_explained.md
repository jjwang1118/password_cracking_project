# DBS_beam_contrastive_search 類詳細說明

## 概述

`DBS_beam_contrastive_search` 是一個結合了 **Beam Search** 和 **Contrastive Search** 機制的密碼生成類別。它通過維護每個 beam 的歷史隱藏狀態（hidden states），並使用對比懲罰（contrastive penalty）來避免生成重複或過於相似的序列，從而提高生成密碼的多樣性和質量。

## 核心設計理念

### 1. 傳統 Beam Search 的局限性
傳統的 beam search 只基於**機率分數**來選擇候選序列，容易產生：
- 重複的 token 模式
- 缺乏多樣性的輸出
- 過度關注高頻詞彙

### 2. Contrastive Search 的改進
Contrastive Search 引入了**對比機制**：
- 追蹤每個序列的生成歷史
- 計算當前候選與歷史的相似度
- 懲罰與歷史過於相似的候選
- 在**機率**與**多樣性**之間取得平衡

---

## 類別結構

### 初始化 (`__init__`)

```python
def __init__(self, info_cache: tuple, vocab_tensor: torch.Tensor, device, initial_hidden: torch.Tensor):
```

#### 參數說明：
- **info_cache**: prompt 的 KV cache（鍵值緩存），用於加速 Transformer 的推理
- **vocab_tensor**: 詞彙表的 tensor，包含所有可用的 token ID
- **device**: 運算設備（CPU/GPU）
- **initial_hidden**: prompt 最後一個 token 的 hidden state，形狀為 `[1, hidden_dim]`

#### 初始化的狀態變量：

| 變量名 | 形狀 | 說明 |
|--------|------|------|
| `info_cache` | tuple | 存儲 prompt 的 KV cache |
| `pw_cache` | tuple/None | 存儲密碼生成過程的 KV cache |
| `pw_idx` | `[1, 0]` | 當前所有 beam 的 token 序列（初始為空）|
| `beam_prob` | `[1, 1]` | 每個 beam 的累積對數機率 |
| `search_prob` | `[1, 1]` | 搜索寬度內的累積對數機率 |
| `vocab_tensor` | `[vocab_size]` | 詞彙表 |
| `acc_hidden_state` | `[1, 1, hidden_dim]` | 每個 beam 的歷史隱藏狀態 |

#### 關鍵設計：
```python
self.acc_hidden_state = initial_hidden.unsqueeze(1)  # [1, 1, hidden_dim]
```
- 初始時只有 1 個 beam
- 歷史長度為 1（只有 prompt 的最後一個隱藏狀態）
- 隨著生成過程，這個 tensor 會沿著 `history_len` 維度增長

---

## 核心方法詳解

### 1. `return_beam_width()`

```python
def return_beam_width(self):
    return self.pw_idx.shape[0]
```

**功能**：返回當前保留的 beam 數量

**實現**：通過 `pw_idx` 的第一維度（batch 維度）獲取

---

### 2. `accumulate_hidden()`

```python
def accumulate_hidden(self, hidden_states: torch.Tensor):
    """
    為每個 beam 累積其對應的隱藏狀態
    
    Args:
        hidden_states: 當前步驟所有 beam 的隱藏狀態 [beam_width, hidden_dim]
    """
    new_hidden = hidden_states.unsqueeze(1)  # [beam_width, hidden_dim] -> [beam_width, 1, hidden_dim]
    self.acc_hidden_state = torch.cat([self.acc_hidden_state, new_hidden], dim=1)
    # 結果: [beam_width, history_len+1, hidden_dim]
```

#### 功能：
累積每個 beam 的歷史隱藏狀態，用於後續的對比計算

#### 處理流程：

1. **輸入**：當前步驟所有 beam 的 hidden state `[beam_width, hidden_dim]`
2. **擴展維度**：在第 2 維插入一個維度 `[beam_width, 1, hidden_dim]`
3. **拼接**：沿歷史維度（dim=1）拼接到 `acc_hidden_state`
4. **輸出**：更新後的歷史 `[beam_width, history_len+1, hidden_dim]`

#### 示例：
```
步驟 0: acc_hidden_state = [1, 1, 768]      (只有 prompt)
步驟 1: 生成 5 個 beam
        新增 hidden = [5, 768]
        累積後 = [5, 2, 768]                 (prompt + 第1個token)
步驟 2: 新增 hidden = [5, 768]
        累積後 = [5, 3, 768]                 (prompt + 2個tokens)
```

---

### 3. `compute_contrastive_penalty()` ⭐ 核心方法

```python
def compute_contrastive_penalty(self, current_hidden: torch.Tensor, beam_indices: torch.Tensor, alpha: float = 0.6):
    """
    基於每個 beam 自己的歷史隱藏狀態計算對比懲罰
    
    Args:
        current_hidden: 當前候選的隱藏狀態 [batch_size, hidden_dim]
        beam_indices: 每個候選對應的父 beam 索引 [batch_size]
        alpha: 對比懲罰權重
        
    Returns:
        penalty: 對比懲罰分數 [batch_size]
    """
```

#### 功能：
計算當前候選序列與其父 beam 歷史的相似度，返回懲罰分數

#### 詳細步驟：

##### 步驟 1：檢查歷史是否存在
```python
if self.acc_hidden_state.shape[1] == 0:
    return torch.zeros(batch_size, device=current_hidden.device)
```
- 如果沒有歷史記錄，返回零懲罰

##### 步驟 2：獲取父 beam 的歷史
```python
beam_histories = self.acc_hidden_state[beam_indices]
# 輸入: acc_hidden_state [beam_width, history_len, hidden_dim]
#      beam_indices [batch_size]
# 輸出: [batch_size, history_len, hidden_dim]
```
- 每個候選從其父 beam 繼承歷史

##### 步驟 3：正規化（Normalization）
```python
current_norm = torch.nn.functional.normalize(current_hidden, p=2, dim=-1)
# [batch_size, hidden_dim]

history_norm = torch.nn.functional.normalize(beam_histories, p=2, dim=-1)
# [batch_size, history_len, hidden_dim]
```
- 使用 L2 正規化將向量標準化
- 確保後續計算的是**餘弦相似度**而非歐式距離

##### 步驟 4：計算餘弦相似度
```python
similarity = torch.bmm(
    current_norm.unsqueeze(1),      # [batch_size, 1, hidden_dim]
    history_norm.transpose(1, 2)    # [batch_size, hidden_dim, history_len]
).squeeze(1)                         # [batch_size, history_len]
```

**批次矩陣乘法（bmm）的計算**：
```
對於第 i 個候選：
    similarity[i, j] = dot_product(current_norm[i], history_norm[i, j])
                     = cosine_similarity(當前hidden, 第j個歷史hidden)
```

##### 步驟 5：取最大相似度
```python
max_similarity = similarity.max(dim=-1)[0]  # [batch_size]
```
- 對每個候選，找出與歷史**最相似**的那個時間步
- 代表「當前候選與歷史的最大重複程度」

##### 步驟 6：計算懲罰
```python
penalty = alpha * max_similarity  # [batch_size]
```
- `alpha`：懲罰強度（預設 0.6）
- 相似度越高，懲罰越大

#### 數學原理：

**懲罰計算**：
$$
\text{penalty}_i = \alpha \cdot \max_{j \in \text{history}} \cos(\mathbf{h}_{\text{current}, i}, \mathbf{h}_{\text{history}, j})
$$

**最終分數計算**：
$$
\text{score}(w_{n+1}) = (1-\alpha) \times \log P(w_{n+1} | w_1 \dots w_n) - \text{penalty}_i
$$

其中：
- $\mathbf{h}_{\text{current}, i}$：第 $i$ 個候選的當前隱藏狀態
- $\mathbf{h}_{\text{history}, j}$：第 $j$ 個歷史隱藏狀態
- $\cos(\cdot, \cdot)$：餘弦相似度
- $\alpha$：權重參數（0.0 ~ 1.0）
  - $(1-\alpha)$：模型機率的權重
  - $\alpha$：對比懲罰的權重

#### 實際效果：
- **高相似度** → **高懲罰** → **降低該候選的選擇機率**
- **低相似度** → **低懲罰** → **保持該候選的機率**
- 鼓勵生成與歷史**不同**的 token，提高多樣性
- **α 值越大**：多樣性越高，但可能犧牲機率

---

### 4. `update_by_prob()` ⭐ 核心更新方法

```python
def update_by_prob(self, beam_width, search_width: int, probs: torch.Tensor, 
                   pw_past_key_values, hidden_states: torch.Tensor = None):
    """
    Update beam state based on probability
    
    Args:
        beam_width: 保留的 beam 數量
        search_width: 搜索寬度（通常 >= beam_width）
        probs: 機率分數 [current_beam_width, vocab_size-1]
               **關鍵**：這是從「父節點視角」出發的機率分布
               每一行代表一個父 beam 對所有詞彙的預測機率
               probs[i, j] = 父 beam i 預測下一個 token 是詞 j 的 log 概率
        pw_past_key_values: KV cache
        hidden_states: 當前步驟的 hidden states [current_beam_width, hidden_dim]
    """
```

#### 功能：
根據機率分數更新 beam 狀態，並管理歷史隱藏狀態

#### 詳細步驟：

##### 步驟 1：計算總機率
```python
tot_probs = (self.beam_prob.reshape(-1, 1) + probs).reshape(-1)
# 輸入: beam_prob [current_beam_width, 1]
#      probs [current_beam_width, vocab_size-1]
# 輸出: tot_probs [current_beam_width * (vocab_size-1)]
```

**關鍵概念：父節點視角**

`probs` 是從**父 beam 的視角**出發的機率分布：
- `probs[0]`：父 beam 0 認為下一個 token 是各個詞的機率
- `probs[1]`：父 beam 1 認為下一個 token 是各個詞的機率
- `probs[i, j]`：父 beam i 預測下一個 token 是詞 j 的 log 概率

這些機率來自於對每個父 beam 執行模型 forward 得到的預測結果。

**PyTorch Broadcasting 機制詳解**：

Broadcasting 規則：
1. 維度相等，或
2. 某一維度為 1（會自動複製），或
3. 某張量缺少該維度（視為 1）

**注意**：如果兩個維度都不是 1 且不相等，會報錯！

```
beam_prob:     [[p1],     →  [[p1, p1, ..., p1],    # p1 複製 vocab_size-1 次
                [p2],         [p2, p2, ..., p2],    # p2 複製 vocab_size-1 次
                [p3]]         [p3, p3, ..., p3]]    # p3 複製 vocab_size-1 次
               [beam_width, 1]

probs:         [[q1, q2, ..., qV],  # 父 beam 0 的預測
                [r1, r2, ..., rV],  # 父 beam 1 的預測
                [s1, s2, ..., sV]]  # 父 beam 2 的預測
               [beam_width, vocab_size-1]

tot_probs = beam_prob + probs
         = [[p1+q1, p1+q2, ..., p1+qV],  # 父beam 0 的所有擴展路徑
            [p2+r1, p2+r2, ..., p2+rV],  # 父beam 1 的所有擴展路徑
            [p3+s1, p3+s2, ..., p3+sV]]  # 父beam 2 的所有擴展路徑
         [beam_width, vocab_size-1]

展平後: [p1+q1, p1+q2, ..., p1+qV, p2+r1, p2+r2, ..., p2+rV, p3+s1, ..., p3+sV]
        [beam_width × (vocab_size-1)]
```

**為什麼形狀總是匹配？**

在正常情況下，`probs.shape[0]` 總是等於 `self.beam_prob.shape[0]`，因為：
1. `probs` 是對當前所有 beam 做 forward 後收集的結果
2. 只收集前 `beam_width` 個 beam 的預測
3. 因此 `probs` 的第一維度 = 當前 `beam_width`

如果形狀不匹配，PyTorch 會拋出 `RuntimeError`。

##### 步驟 2：選擇 Top-K 候選
```python
self.search_prob, search_idx = torch.topk(tot_probs, search_width, largest=True)
self.beam_prob, self.beam_idx = torch.topk(tot_probs, beam_width, largest=True)

self.search_prob = self.search_prob.reshape(-1, 1)  # [search_width, 1]
self.beam_prob = self.beam_prob.reshape(-1, 1)      # [beam_width, 1]
```

- **search_width**：用於前向計算的候選數（可能大於 beam_width）
- **beam_width**：最終保留的候選數
- **search_idx** / **beam_idx**：選中的候選在 `tot_probs` 中的索引

##### 步驟 3：解碼索引（反向映射）
```python
self.last_beam_index = search_idx // (self.vocab_tensor.shape[0] - 1)
word_index = torch.remainder(search_idx, self.vocab_tensor.shape[0] - 1)
```

**反向映射的數學原理**

在步驟 1 中，我們將 2D 的機率矩陣 `[beam_width, vocab_size-1]` 展平成 1D 的 `tot_probs`。  
現在需要從 1D 索引**反推回** 2D 座標 (beam_idx, token_idx)。

**Row-Major 順序（列優先）**：
- 展平公式：`flat_idx = row * num_cols + col`
- 反推公式：
  - `row = flat_idx // num_cols` （整數除法）
  - `col = flat_idx % num_cols`  （取餘數）

**具體例子**：

假設 `vocab_size = 4`，去掉 EOS 後剩 3 個詞，`beam_width = 2`

```
原始 2D 矩陣（tot_probs 展平前）:
          詞0   詞1   詞2
   beam0 [ 0    1    2 ]
   beam1 [ 3    4    5 ]

展平後 1D (tot_probs):
   [0, 1, 2, 3, 4, 5]

反向映射（num_cols = vocab_size - 1 = 3）:
   idx=0: row=0//3=0, col=0%3=0 → beam 0, 詞 0
   idx=1: row=1//3=0, col=1%3=1 → beam 0, 詞 1
   idx=2: row=2//3=0, col=2%3=2 → beam 0, 詞 2
   idx=3: row=3//3=1, col=3%3=0 → beam 1, 詞 0
   idx=4: row=4//3=1, col=4%3=1 → beam 1, 詞 1
   idx=5: row=5//3=1, col=5%3=2 → beam 1, 詞 2
```

**實際應用**：
```
假設 vocab_size = 100, search_idx = [205, 312, 107]
num_cols = vocab_size - 1 = 99

last_beam_index = [205//99, 312//99, 107//99] = [2, 3, 1]  (父beam索引)
word_index = [205%99, 312%99, 107%99] = [7, 15, 8]         (詞彙索引)

含義：
  - 候選 1：來自第 2 個 beam，選擇第 7 個詞彙
  - 候選 2：來自第 3 個 beam，選擇第 15 個詞彙
  - 候選 3：來自第 1 個 beam，選擇第 8 個詞彙
```

**為什麼減 1？** 
因為 `probs` 不包含 EOS token（在訓練時通常將 EOS 放在詞彙表最後），所以實際的矩陣寬度是 `vocab_size - 1`。

##### 步驟 4：記錄 beam_width 內的父 beam 索引
```python
self.beam_parent_index = self.beam_idx // (self.vocab_tensor.shape[0] - 1)
```
- 用於後續重排歷史隱藏狀態

##### 步驟 5：更新序列（構建新的候選路徑）
```python
self.pw_idx = torch.cat([
    self.pw_idx[self.last_beam_index],           # 繼承父beam的序列
    self.vocab_tensor[word_index].reshape(-1, 1) # 添加新token
], dim=1)
```

**功能**：通過拼接父序列和新 token 來構建下一層的候選序列

**詳細流程**：

1. **`self.pw_idx[self.last_beam_index]`**：選擇父序列
   - 形狀變化：`[beam_width, seq_len]` → `[search_width, seq_len]`
   - 根據 `last_beam_index` 挑選對應的父 beam 序列

2. **`self.vocab_tensor[word_index].reshape(-1, 1)`**：獲取新 token
   - 形狀變化：`[search_width]` → `[search_width, 1]`
   - 將選中的詞彙 ID 整形成列向量

3. **`torch.cat([...], dim=1)`**：沿序列長度維度拼接
   - 結果形狀：`[search_width, seq_len+1]`

**完整示例**：
```
當前狀態:
  pw_idx (舊):        [[1, 2],      # beam 0: 序列 [1, 2]
                       [3, 4]]       # beam 1: 序列 [3, 4]
  
  last_beam_index:    [1, 0, 1]     # 新候選的父 beam 索引
  word_index:         [5, 6, 7]     # 新候選選擇的 token
  vocab_tensor:       [10, 20, 30, 40, 50, 60, 70, 99]  # 詞彙表

執行過程:
  1. 選擇父序列:
     pw_idx[last_beam_index] = [[3, 4],   # 來自 beam 1
                                 [1, 2],   # 來自 beam 0
                                 [3, 4]]   # 來自 beam 1
  
  2. 獲取新 token:
     vocab_tensor[word_index].reshape(-1, 1) = [[60],   # token_id=60
                                                 [70],   # token_id=70
                                                 [99]]   # token_id=99
  
  3. 拼接結果:
     pw_idx (新) = [[3, 4, 60],   # 候選 0: 繼承 beam 1 + 新增 token 60
                    [1, 2, 70],   # 候選 1: 繼承 beam 0 + 新增 token 70
                    [3, 4, 99]]   # 候選 2: 繼承 beam 1 + 新增 token 99
```

**核心邏輯**：
這行代碼實現了 **Beam Search 的樹擴展**：
- 從當前層的 beam 中選擇父節點（通過 `last_beam_index`）
- 為每個父節點追加一個新 token（通過 `word_index`）
- 生成下一層的候選序列

每一步都在「繼承」最優路徑的同時探索新的分支。

##### 步驟 6：更新 KV cache
```python
self.pw_cache = pw_past_key_values
```

##### 步驟 7：更新歷史隱藏狀態
```python
if hidden_states is not None:
    # 重排歷史：選中的 beam 繼承其父 beam 的歷史
    self.acc_hidden_state = self.acc_hidden_state[self.beam_parent_index]
    # [beam_width, history_len, hidden_dim]
    
    # 取出被選中的 beam 對應的 hidden states 並累積
    selected_hidden = hidden_states[self.beam_parent_index]  # [beam_width, hidden_dim]
    self.accumulate_hidden(selected_hidden)
else:
    # 第一層：沒有新的 hidden states，只需廣播初始歷史到所有 beam
    self.acc_hidden_state = self.acc_hidden_state.expand(beam_width, -1, -1).clone()
    # [1, 1, hidden_dim] -> [beam_width, 1, hidden_dim]
```

**歷史重排示例**：
```
假設：
  beam_width = 3
  當前 acc_hidden_state = [[h1_0, h1_1],    # beam 0 的歷史
                            [h2_0, h2_1],    # beam 1 的歷史
                            [h3_0, h3_1]]    # beam 2 的歷史
  
  beam_parent_index = [1, 0, 1]  # 新的3個beam分別來自舊的beam 1, 0, 1

重排後：
  acc_hidden_state = [[h2_0, h2_1],    # 繼承beam 1的歷史
                      [h1_0, h1_1],    # 繼承beam 0的歷史
                      [h2_0, h2_1]]    # 繼承beam 1的歷史

累積新的hidden state後：
  acc_hidden_state = [[h2_0, h2_1, h_new_0],
                      [h1_0, h1_1, h_new_1],
                      [h2_0, h2_1, h_new_2]]
```

---

## 完整工作流程

### 初始化階段
```
1. 創建 beam 對象，初始狀態：
   - 1 個 beam
   - 空的序列
   - prompt 的 hidden state 作為歷史起點
```

### 生成循環（每一步）

#### 階段 1：前向計算（在 `contrastive_search` 函數中）
```python
for i in range(forward_num):
    # 1. 模型前向傳播，獲取 logits 和 hidden states
    outputs = model.forward(...)
    
    # 2. 計算對比懲罰
    current_hidden = outputs.hidden_states[-1][:, -1, :]
    penalty = beam.compute_contrastive_penalty(current_hidden, current_beam_indices, alpha)
    
    # 3. 應用懲罰到機率分數（論文公式）
    # score = (1-α) × log_prob - α × max_similarity
    batch_word_probs[:, :-1] = (1 - alpha) * batch_word_probs[:, :-1] - penalty.unsqueeze(1)
    
    # 4. 收集 hidden states
    hidden_states_batch.append(current_hidden)
```

#### 階段 2：更新 beam 狀態
```python
# 合併所有 batch 的 hidden states
all_hidden = torch.cat(hidden_states_batch, dim=0)

# 更新 beam，包含歷史重排和累積
beam.update_by_prob(next_beam_width, next_reserve_width, 
                    word_probs, pw_past_key_values, all_hidden)
```

### 流程圖
```
┌─────────────────────────────────────────────────────────────┐
│ 步驟 t: 有 N 個 beam，每個 beam 有長度為 t 的歷史         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │ 1. 對每個 beam 擴展 vocab_size 個 │
        │    候選（總共 N × vocab_size 個）  │
        └──────────┬───────────────────────┘
                   │
                   ▼
        ┌──────────────────────────────────┐
        │ 2. 模型前向計算 → 獲取 hidden state│
        └──────────┬───────────────────────┘
                   │
                   ▼
        ┌──────────────────────────────────┐
        │ 3. 計算每個候選與其父 beam 歷史的  │
        │    最大餘弦相似度 → penalty       │
        └──────────┬───────────────────────┘
                   │
                   ▼
        ┌──────────────────────────────────┐
        │ 4. 調整機率：                     │
        │    adjusted_prob = log_prob - penalty │
        └──────────┬───────────────────────┘
                   │
                   ▼
        ┌──────────────────────────────────┐
        │ 5. 選擇 Top-beam_width 個候選    │
        └──────────┬───────────────────────┘
                   │
                   ▼
        ┌──────────────────────────────────┐
        │ 6. 重排歷史：選中的 beam 繼承其   │
        │    父 beam 的歷史                 │
        └──────────┬───────────────────────┘
                   │
                   ▼
        ┌──────────────────────────────────┐
        │ 7. 累積新的 hidden state 到歷史   │
        └──────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│ 步驟 t+1: 有 beam_width 個 beam，歷史長度增加到 t+1         │
└──────────────────────────────────────────────────────────────┘
```

---

## 與傳統 Beam Search 的對比

| 特性 | 傳統 Beam Search | DBS_beam_contrastive_search |
|------|------------------|------------------------------|
| **選擇標準** | 僅基於機率 | 機率 - 對比懲罰 |
| **歷史追蹤** | 無 | 追蹤每個 beam 的 hidden states |
| **多樣性** | 低（容易重複） | 高（避免與歷史相似） |
| **計算開銷** | 低 | 中（需計算相似度） |
| **記憶體使用** | 低 | 高（需存儲歷史） |
| **適用場景** | 通用文本生成 | 需要高多樣性的生成任務（如密碼生成） |

---

## 參數調整指南

### `alpha`（對比懲罰強度）
- **預設值**：0.6
- **範圍**：[0, 1]
- **效果**：
  - `alpha = 0`：退化為傳統 beam search
  - `alpha → 1`：強烈懲罰重複，極高多樣性
  - **推薦**：0.5 - 0.7

### `beam_width`
- **說明**：保留的候選數量
- **效果**：
  - 太小：可能錯過好的候選
  - 太大：計算開銷大，記憶體消耗高
  - **推薦**：50 - 200（取決於任務）

#### Beam Width 的動態變化規則

**同一層內**：`beam_width` 固定不變
- 在同一生成步驟中，始終保持指定的 `beam_width` 數量
- `probs.shape[0]` 總是等於 `self.beam_prob.shape[0]`（當前層的 `beam_width`）

**不同層之間**：`beam_width` 可以改變
```python
beam_width_list = [2, 5, 3, 10]  # 每層的 beam 數量不同

Layer 0: 2 個 beam
Layer 1: 5 個 beam  # ✅ 增加了
Layer 2: 3 個 beam  # ✅ 減少了
Layer 3: 10 個 beam # ✅ 又增加了
```

**關鍵約束**：
所有新 beam 都必須從**現有 beam 擴展**而來，不存在「憑空新增路徑」。

每層的 `beam_width` 受限於父層的候選數量：
```python
max_candidates = beam_width[i-1] × (vocab_size - 1)
beam_width[i] = min(beam_width[i], max_candidates)
```

在 `contrastive_search` 函數中有自動限制邏輯：
```python
# 第 0 層：最多 vocab_size-1 個候選（去掉 EOS）
beam_width_list[0] = min(beam_width_list[0], vocab_size - 1)

# 第 i 層：最多 beam_width[i-1] × (vocab_size-1) 個候選
for i in range(1, len(beam_width_list)):
    max_candidates = beam_width_list[i-1] * (vocab_size - 1)
    beam_width_list[i] = min(beam_width_list[i], max_candidates)
```

### `search_width`
- **說明**：前向計算的候選數（通常 ≥ beam_width）
- **用途**：在計算懲罰前考慮更多候選
- **推薦**：`beam_width × 1.5 - 2`

---

## 優化建議

### 記憶體優化
```python
# 限制歷史長度（只保留最近 K 步）
if self.acc_hidden_state.shape[1] > max_history_len:
    self.acc_hidden_state = self.acc_hidden_state[:, -max_history_len:, :]
```

### 計算優化
```python
# 使用較低精度（float16）
self.acc_hidden_state = self.acc_hidden_state.half()
```

### 批次優化
```python
# 將相似度計算移到 GPU
# 確保所有 tensor 都在同一設備上
```

---

## 實際應用案例

### 密碼生成
```python
# 生成多樣且高機率的密碼
passwords = contrastive_search(
    model=model,
    input_ids=prompt_ids,
    batch_size=32,
    beam_width_list=[100] * 12,  # 12個字符的密碼
    vocab=vocab,
    use_contrastive=True,
    contrastive_alpha=0.6
)
```

### 為什麼適合密碼生成？
1. **避免常見模式**：對比機制懲罰重複的字符序列
2. **保持高機率**：不完全拋棄機率信息
3. **平衡多樣性與可能性**：生成既多樣又符合密碼規律的候選

---

## 總結

`DBS_beam_contrastive_search` 類通過以下三個核心機制實現了高質量的序列生成：

1. **歷史追蹤**：維護每個 beam 的 hidden state 歷史
2. **對比計算**：基於餘弦相似度計算懲罰
3. **動態更新**：每步更新時重排歷史並累積新狀態

這種設計在**機率**和**多樣性**之間取得了良好的平衡，特別適合需要避免重複模式的生成任務。

---

## 參考文獻

- **Contrastive Search**: Su et al. (2022) - "A Contrastive Framework for Neural Text Generation"
- **Beam Search**: Graves (2012) - "Sequence Transduction with Recurrent Neural Networks"

---

**文檔版本**：1.1  
**最後更新**：2026-01-12  
**修正內容**：更正 Contrastive Search 公式，使用 (1-α)×log_prob - α×penalty 的正確權重分配  
**作者**：PassLLM 項目組
