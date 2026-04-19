FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Cache directories — all models are pre-downloaded during build/setup, not at runtime
ENV TORCH_HOME=/app/.cache/torch
ENV MODELSCOPE_CACHE=/app/.cache/modelscope
ENV HF_HOME=/app/.cache/huggingface

# Offline mode — prevents any internet calls during inference
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV MODELSCOPE_ENVIRONMENT=local

# System deps + Python 3.12 (via deadsnakes PPA — not in Ubuntu 22.04 default repos)
RUN apt-get update && apt-get install -y software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3.12-venv \
    ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

# Bootstrap pip for Python 3.12 (distutils removed in 3.12, system pip won't work)
RUN python3.12 -m ensurepip --upgrade && \
    python3.12 -m pip install --upgrade pip

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.12 1

WORKDIR /app

# Install PyTorch with CUDA 12.4
RUN pip install --no-cache-dir \
    torch==2.6.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install all other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Copy source code
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY infer.py .

# Pre-download ModelScope backbone models (CAM++ and ERes2Net architecture + pretrained weights)
# This runs WITH internet during docker build — cached in image layer
RUN python3 -c "\
from modelscope.pipelines import pipeline; \
from modelscope.utils.constant import Tasks; \
pipeline(task=Tasks.speaker_verification, model='damo/speech_campplus_sv_en_voxceleb_16k', model_revision='v1.0.2'); \
pipeline(task=Tasks.speaker_verification, model='damo/speech_eres2net_sv_en_voxceleb_16k', model_revision='master'); \
print('ModelScope models cached OK')"

# Weights directory (populated in Step 1 before offline inference)
RUN mkdir -p weights embeddings

ENTRYPOINT ["python3", "infer.py"]
CMD ["--help"]
