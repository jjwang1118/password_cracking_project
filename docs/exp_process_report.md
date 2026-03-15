# 實驗流程報告

## 1. Data Clean（資料清洗）

1. 刪除 non-ASCII char
2. 刪除非 Password map 的 char（參考 PassIlm）
3. 刪除長度 < 8 的 Password
4. 刪除重複資料

---

## 2. Stastic（統計分析）

1. 統計 domain、length、org 分布/統計
2. Delete value（廢棄 / 清理殘餘資訊）
3. 設計斷點

---

## 3. 訓練 Model

- **Random Seed**
- **Model Name**: Llama-3-8b
- **微調方式**: LoRA
- **資料量**: 選取 50% of data
  - 所有pair_sister 數量 > 1 的資料
  - 其他資料隨機選取

---

## 4. Generate Algorithm（生成演算法）

- **Beam Search** + **Contrastive Search**

---

## 5. ML — Control Generative（控制生成）

- **需要的內容**：
  - header 數
  - 虛擬token seq
  - dimension/header

-  **定義安全/不安全的密碼**:

    1. pairs sister count > 1 ， 帳號和密碼相似度
    2. 丟入其他模型看破解情況，如果有被猜到則是不安全的密碼
    3. 其他定義

- 使用訓練lm的資料
---



