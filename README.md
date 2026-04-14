# Password Cracking Project

A LLM-based password guessing framework that fine-tunes Llama models for both trawling and targeted password attacks, with prefix-tuning for controllable generation (secure / unsafe passwords).

---

## Project Structure

```
.
├── train.py                  # LoRA fine-tuning (trawling / targeted attack)
├── train_prefix.py           # Prefix-tuning for controllable generation
├── eval.py                   # Model evaluation & password generation
├── label.py                  # Label passwords as safe / unsafe
├── process_data.py           # Data preprocessing pipeline
├── split_data.py             # Train/test split
├── stastic_data.py           # Dataset statistics
├── config.yaml               # All hyperparameters & paths
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── util/
│   ├── generate_control.py   # Prefix-tuning model (llamaModel_control)
│   ├── datacollector.py      # Dataset class
│   ├── tokenize.py           # Tokenization utilities
│   ├── prompt_template.py    # Prompt templates
│   ├── search.py             # Beam / contrastive search
│   └── label_safe_unsafe_pw.py
├── data_process/             # Data processing scripts
├── datasets_filtered/        # Processed datasets
│   ├── labeled/              # train_data.csv / test_data.csv
│   └── split/                # train_data.jsonl / test_data.jsonl
├── model/                    # Base model weights (Llama-3.2-3B-Instruct)
├── checkpoints/              # Saved checkpoints
└── docs/                     # Architecture & experiment notes
```

---

## Setup

### Requirements

- Python 3.10+
- CUDA 13.0+
- PyTorch 2.9.0+cu130

```bash
# Install PyTorch (CUDA 13.0)
pip install torch==2.9.0+cu130 --index-url https://download.pytorch.org/whl/cu130

# Install other dependencies
pip install -r requirements.txt
```

### Docker (Recommended)

```bash
docker compose up --build
```

---

## Usage

### 1. Data Preprocessing

```bash
python process_data.py
python stastic_data.py
python split_data.py
```

### 2. Label Passwords (Safe / Unsafe)

```bash
python label.py
```

Outputs `datasets_filtered/labeled/train_data.csv` and `test_data.csv` with a `safe_label` column (0 = safe, 1 = unsafe).

### 3. LoRA Fine-tuning (Trawling / Targeted Attack)

```bash
python train.py
```

Configure model, LoRA, and training parameters in `config.yaml` under `train:`.

### 4. Prefix-tuning (Controllable Generation)

```bash
python train_prefix.py
```

Trains two sets of prefix parameters (secure / unsafe) injected into every Transformer layer's KV cache. Configure under `label:` in `config.yaml`.

Checkpoints saved to `checkpoints/{model_name}/prefix/checkpoint_epoch_{n}.pt`.

### 5. Evaluation

```bash
python eval.py
```

---

## Model Architecture

### LoRA Fine-tuning

Standard parameter-efficient fine-tuning targeting `q_proj`, `k_proj`, `v_proj` with rank `r=16`.

### Prefix-tuning (`llamaModel_control`)

Injects learnable KV prefix tensors into every Transformer layer:

```
parameterlist: 2 (secure/unsafe) × num_layers × 2 (k/v)
               shape per param: (num_kv_heads, prefix_len, head_dim)
```

Training loss combines three objectives:

| Loss | Weight | Purpose |
|------|--------|---------|
| LM loss (cross-entropy) | 1 | Generate correct passwords under correct control |
| Contrastive loss (−NLL) | 4 | Suppress generation under wrong control |
| KL loss | 1.6 | Preserve base model behaviour without prefix |

### Generation

Supports **beam search** and **contrastive search** (configurable in `config.yaml` under `search:`).

---

## Configuration (`config.yaml`)

Key sections:

| Section | Description |
|---------|-------------|
| `train` | LoRA fine-tuning settings |
| `eval` | Generation & evaluation settings |
| `label` | Prefix-tuning & labeling settings |
| `search` | Search algorithm settings |

---

## Dataset

Based on the `CompilationOfManyBreaches` dataset, filtered and processed:

1. Remove non-ASCII characters
2. Remove characters outside printable password charset
3. Remove passwords shorter than 8 characters
4. Remove duplicates

---

## References

- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)
- [SVEN: Security and Vulnerability-aware Code Generation](https://arxiv.org/abs/2302.09205)
- [PassLLM](https://github.com/passlm/passlm)
