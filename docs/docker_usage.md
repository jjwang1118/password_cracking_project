# Docker 使用說明

## 前置需求

- [Docker](https://docs.docker.com/get-docker/) >= 29.0
- [Docker Compose](https://docs.docker.com/compose/) >= 2.0
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

確認 NVIDIA Container Toolkit 已正確安裝：

```bash
nvidia-smi
docker run --rm --runtime=nvidia nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

---

## 專案結構與 Volume 說明

大型資料夾**不會打包進 image**，透過 volume 掛載方式使用：

| 本機目錄 | 容器內路徑 | 說明 |
|---------|-----------|------|
| `./model/` | `/workspace/model` | 預訓練模型（唯讀） |
| `./datasets_filtered/` | `/workspace/datasets_filtered` | 資料集 |
| `./checkpoints/` | `/workspace/checkpoints` | 訓練輸出的 checkpoint |
| `./logs/` | `/workspace/logs` | TensorBoard log |
| `./gen/` | `/workspace/gen` | eval 輸出結果 |

---

## 建立 Image

```bash
# 第一次使用，或 Dockerfile / requirements.txt 有變更時執行
docker compose build
```

> 首次 build 需下載 CUDA base image 與安裝 PyTorch，約需 10～20 分鐘。

---

## 執行各任務

### 訓練（train.py）

```bash
docker compose run --rm train
```

訓練結果（checkpoint）會輸出至本機 `./checkpoints/` 目錄，TensorBoard log 輸出至 `./logs/`。

### 評估（eval.py）

```bash
docker compose run --rm eval
```

評估結果輸出至本機 `./gen/` 目錄。

### 標記（label.py）

```bash
docker compose run --rm label
```

標記結果輸出至本機 `./datasets_filtered/labeled/`。

---

## 互動式 Shell 除錯

```bash
# 進入容器內的 bash
docker compose run --rm train bash

# 容器內可直接執行任意 python 指令
python train.py
python -c "import torch; print(torch.cuda.is_available())"
```

---

## TensorBoard 監控訓練

訓練過程中，在另一個終端啟動 TensorBoard 並掛載 log 目錄：

```bash
docker run --rm -p 6006:6006 \
  -v $(pwd)/logs:/logs \
  tensorflow/tensorflow:latest \
  tensorboard --logdir /logs --host 0.0.0.0
```

瀏覽器開啟 `http://localhost:6006`。

---

## 常用維護指令

```bash
# 查看建立的 image
docker images | grep pw-cracking

# 清除停止的容器
docker compose down

# 強制重新 build（不使用快取）
docker compose build --no-cache

# 刪除 image
docker rmi pw-cracking:latest
```

---

## 設定檔修改

所有超參數統一在 [`config.yaml`](../config.yaml) 調整，不需要重新 build image。
修改 `config.yaml` 後直接執行對應的 `docker compose run` 即可生效。
