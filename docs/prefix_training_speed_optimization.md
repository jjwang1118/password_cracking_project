# Prefix-Tuning 訓練速度優化清單

## 優先級高（改動極低，立即可做）

- [ ] **啟用 Flash Attention 2**
  - 在 `train_prefix.py` 載入模型時加入 `attn_implementation="flash_attention_2"` 與 `torch_dtype=torch.bfloat16`
  - 預期加速：20~40%
  - 參考論文：FlashAttention-2 (ICLR 2024, arXiv:2307.08691)

## 優先級中（改動低，效益明確）

- [ ] **KL Forward 降頻**
  - 目前每個 batch 做 3 次 forward（LM / Contrastive / KL），改為每 N step 才做一次 KL forward
  - 預期加速：25~33%
  - 改動位置：`util/generate_control.py` → `train_epoch()`

- [ ] **增大 effective batch size**
  - 目前 `batch_size=1, grad_acc=2`（effective=2），增大 `grad_acc` 至 8~16
  - 減少 optimizer step 次數，提升 GPU 利用率
  - 改動位置：`config.yaml` → `label.grad_acc`

- [ ] **LLaMA-Adapter Zero-init Gate**
  - Prefix 初始影響為零，收斂更快，減少所需 epoch 數
  - 參考論文：LLaMA-Adapter (ICLR 2024, arXiv:2303.16199)
  - 改動位置：`util/generate_control.py` → `__init__()` 加 per-layer learnable gate

## 優先級低（改動高，需重構 forward）

- [ ] **GradCache（Contrastive Learning 梯度緩存）**
  - 合併 correct + wrong control 的 forward，省掉一個完整 backward graph
  - 預期效益：省 GPU memory → 可換更大 batch
  - 參考論文：GradCache (EMNLP 2021)
  - 實作：[github.com/luyug/GradCache](https://github.com/luyug/GradCache)

- [ ] **Prefix Propagation（跨層共享 Prefix）**
  - 目前 28層 × 2 control × 2(k/v) = 112 個 prefix 向量，改為只在第一層注入，後續層自然 propagate
  - 大幅降低 `parameterlist` 大小與 attention 計算量
  - 參考論文：Prefix Propagation (arXiv:2303.08518)
  - 改動位置：`util/generate_control.py` → `get_past_from_prefix()` 與 `forward()`

## 備註

- 上述優化互相獨立，可依序疊加
- Flash Attention 2 + KL 降頻 是最快見效的組合，建議優先實施
- Prefix Propagation 會降低少量表達能力，需實驗驗證效果後再決定是否採用
