# DreamZero + Wan2.2: Python 3.11, PyTorch 2.8 (CUDA 12), flash_attn last via wheel
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_VERSION=cu129 \
    TORCH_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu129"

# System deps for video, builds, Wan2.2 tooling
# Note: newer Debian images ship `libgl1` instead of `libgl1-mesa-glx`.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/dreamzero

COPY pyproject.toml README.md ./
COPY . .

# Core DreamZero + Wan2.2 deps (PyTorch 2.8, cu129)
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -e . --extra-index-url "${TORCH_EXTRA_INDEX_URL}"

# Wan2.2 extras: librosa, decord (decord2 is in pyproject)
RUN pip install librosa decord

# Install flash_attn last from prebuilt wheel (Python 3.11, CUDA 12, PyTorch 2.8)
ARG FLASH_ATTN_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
RUN pip install "${FLASH_ATTN_WHEEL_URL}"

ENV PYTHONPATH=/workspace/dreamzero:${PYTHONPATH}

CMD ["bash"]
