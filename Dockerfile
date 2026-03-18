# PyTorch 2.2.2 + CUDA 12.1 + cuDNN 8 — Python 3.10 and pip included
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

EXPOSE 8014

CMD ["python", "main.py"]
