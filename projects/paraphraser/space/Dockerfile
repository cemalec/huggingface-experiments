FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    "transformers>=4.40.0" \
    "datasets>=2.20.0" \
    "trl>=1.6.0" \
    "peft>=0.10.0" \
    "huggingface_hub>=0.23.0" \
    "accelerate>=0.30.0"

COPY train.py run_training.py ./

EXPOSE 7860

CMD ["python", "run_training.py"]
