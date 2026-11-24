

FROM python:3.11-slim


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4

WORKDIR /app

# System libraries required by OpenCV and inference packages

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        -o Acquire::Retries=5 \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*


# Install Python dependencies
COPY requirements-docker.txt .

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install --no-cache-dir -r requirements-docker.txt

# Copy application, configuration and models
COPY . .

# Writable application output
RUN mkdir -p /app/output

# Run ACT
CMD ["python", "src/ACT.py"]
