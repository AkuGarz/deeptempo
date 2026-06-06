# DeepTempo development Docker image
# GPU support requires nvidia-container-toolkit on the host.
#
# Build:
#   docker build -t deeptempo:dev -f docker/Dockerfile .
#
# Run:
#   docker run --gpus all -it -v $(pwd):/workspace deeptempo:dev

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Symlink python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /workspace

# Install PyTorch with CUDA
RUN pip install --no-cache-dir \
    torch==2.1.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy only deps first for caching
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev,notebooks]"

# Copy the rest
COPY . .

RUN pip install --no-cache-dir -e ".[dev,notebooks]"

CMD ["python", "-m", "pytest", "tests/", "-v"]
