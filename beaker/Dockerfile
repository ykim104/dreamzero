# DreamZero + Wan2.2: base has CUDA 12.8 + Ubuntu 22.04; we install PyTorch 2.8 and deps
FROM ghcr.io/allenai/cuda:12.8-dev-ubuntu22.04-torch2.7.1-v1.2.199 AS conda_env_builder

# DeepSpeed requires CUDA_HOME at import time (must be set in image)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu128" \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# DreamZero requires Python 3.11 (flash_attn wheel is cp311); base may be 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m ensurepip --upgrade && python3.11 -m pip install --upgrade pip setuptools wheel

# Default python/python3 to 3.11 so scripts and torchrun use it
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Pin NumPy 1.26.x first so PyTorch and all deps use it (avoids numpy._core / _signature_descriptor errors in multiprocessing)
RUN python3.11 -m pip install "numpy==1.26.4"

# Override base torch with 2.8 (Python 3.11, keeps CUDA_HOME for DeepSpeed)
RUN python3.11 -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --extra-index-url "${TORCH_EXTRA_INDEX_URL}"

WORKDIR /root/dreamzero

COPY pyproject.toml README.md ./
COPY . .

# DreamZero + Wan2.2 deps (torch already 2.8, Python 3.11)
# --ignore-installed blinker: base image has distutils-installed blinker; pip cannot uninstall it
RUN python3.11 -m pip install -e . --extra-index-url "${TORCH_EXTRA_INDEX_URL}" --ignore-installed blinker

# Wan2.2 extras: librosa, decord (decord2 is in pyproject)
RUN python3.11 -m pip install librosa decord

# Install flash_attn last from prebuilt wheel (Python 3.11, CUDA 12, PyTorch 2.8)
ARG FLASH_ATTN_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
RUN python3.11 -m pip install "${FLASH_ATTN_WHEEL_URL}"

# Ensure NumPy stays 1.26.4 and reinstall scipy/sklearn so they link to it (avoids numpy._core.multiarray failed to import in workers)
RUN python3.11 -m pip install "numpy==1.26.4" --force-reinstall \
    && python3.11 -m pip install --force-reinstall --no-deps scipy scikit-learn

ENV PYTHONPATH=/root/dreamzero:${PYTHONPATH}

CMD ["bash"]
