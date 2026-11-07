FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Скачиваем sample.py из репозитория Viggle-Remake
RUN mkdir -p /app/inference && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Viggle/Viggle-Remake', filename='inference/sample.py', local_dir='/app', local_dir_use_symlinks=False)" && \
    mv /app/Viggle-Remake/inference/sample.py /app/inference/sample.py 2>/dev/null || true

COPY handler.py .

# Сетевой том с весами монтируй при создании эндпоинта:
# --volume viggle-weights:/workspace/weights

CMD ["python", "handler.py"]
