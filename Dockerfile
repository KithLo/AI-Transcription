FROM python:3.11-slim-bookworm

# Install system dependencies (ffmpeg is required by funasr/torchaudio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU PyTorch and requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Environment variables for persistent storage and CPU thread bounding
ENV HF_HOME=/data/cache/huggingface
ENV MODELSCOPE_CACHE=/data/cache/modelscope
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV TORCH_NUM_THREADS=4
ENV PYTHONUNBUFFERED=1

COPY app.py .

EXPOSE 7860

# Run with 'nice -n 15' to lower CPU scheduling priority
CMD ["nice", "-n", "15", "python", "app.py"]
