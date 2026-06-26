FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install torch 2.6+ first (separate layer so it caches independently)
RUN pip3 install --no-cache-dir \
    "torch>=2.6.0" \
    --extra-index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir \
    "transformers>=4.40.0" \
    "datasets>=2.20.0" \
    "trl>=1.6.0" \
    "peft>=0.10.0" \
    "huggingface_hub>=0.23.0" \
    "accelerate>=0.30.0"

WORKDIR /app
COPY train.py run_training.py ./

EXPOSE 7860
CMD ["python3", "run_training.py"]
