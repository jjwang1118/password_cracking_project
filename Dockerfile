# ---------------------------------------------------------------
# Base image: NVIDIA CUDA 13.0 + cuDNN runtime on Ubuntu 24.04
# 若 13.0 尚未發布，可改用 nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
# 並調整下方 PyTorch 安裝指令的 --index-url 為 cu126
# ---------------------------------------------------------------
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---------------------------------------------------------------
# 系統套件 + Python 3.13
# Ubuntu 24.04 內建 Python 3.12，透過 deadsnakes PPA 安裝 3.13
# ---------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.13 \
        python3.13-dev \
        python3.13-venv \
    && rm -rf /var/lib/apt/lists/*

# 設定 python / pip 預設指向 3.13
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.13 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13

WORKDIR /workspace

# ---------------------------------------------------------------
# 安裝 PyTorch 2.9.0 (CUDA 13.0 wheel) — 單獨一層方便快取
# ---------------------------------------------------------------
RUN pip install \
    torch==2.9.0+cu130 \
    torchvision==0.24.0+cu130 \
    torchaudio==2.9.0+cu130 \
    --index-url https://download.pytorch.org/whl/cu130

# ---------------------------------------------------------------
# 安裝其他相依套件
# ---------------------------------------------------------------
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------------------------------------------------------------
# 複製專案程式碼（大型資料/模型透過 volume 掛載，不打包進 image）
# ---------------------------------------------------------------
COPY config.yaml .
COPY train.py eval.py label.py process_data.py split_data.py stastic_data.py ./
COPY util/       ./util/
COPY data_process/ ./data_process/
COPY docs/       ./docs/

# ---------------------------------------------------------------
# 建立 volume 掛載點目錄
# ---------------------------------------------------------------
RUN mkdir -p model datasets_filtered checkpoints logs gen

# ---------------------------------------------------------------
# 預設指令（可被 docker-compose / docker run 覆蓋）
# ---------------------------------------------------------------
CMD ["python", "train.py"]
