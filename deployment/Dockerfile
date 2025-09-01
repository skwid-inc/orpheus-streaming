FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    PATH=/usr/local/cuda/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4 support
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 \
    torchvision==0.21.0

# Install base dependencies
RUN pip install \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.32.0" \
    transformers==4.51.0 \
    python-dotenv \
    pydantic \
    httptools \
    tqdm \
    setuptools \
    snac \
    batched \
    g2p_en \
    aiohttp \
    requests

# Install TensorRT and TensorRT-LLM
RUN pip install tensorrt==10.9.0.34 tensorrt_llm==0.19.0

# Pin cuda-python to avoid import issues
RUN pip install --force-reinstall cuda-python==12.6.0

# Download required NLTK data for g2p_en
RUN python3 -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/outputs /workspace/hf

# Set base environment variables
ENV HF_HOME=/workspace/hf \
    TRANSFORMERS_OFFLINE=0

# MPI and TensorRT-LLM configuration
ENV OMPI_MCA_mpi_yield_when_idle=1 \
    OMPI_MCA_btl_vader_single_copy_mechanism=none \
    OMPI_MCA_mpi_warn_on_fork=0 \
    OMPI_MCA_btl="^openib" \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_MCA_plm_rsh_agent="sh -c" \
    OMPI_MCA_orte_tmpdir_base="/tmp" \
    OMPI_MCA_btl_tcp_if_exclude="lo,docker0" \
    WORLD_SIZE=1 \
    RANK=0 \
    LOCAL_RANK=0 \
    CUDA_VISIBLE_DEVICES=0 \
    NCCL_P2P_DISABLE=1 \
    NCCL_IB_DISABLE=1 \
    TRTLLM_DISABLE_MPI=1 \
    TENSORRT_LLM_USE_MPI=0 \
    TRTLLM_SINGLE_WORKER=1

# Expose port (default 9090, can be overridden by PORT env var)
EXPOSE 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD sh -c 'wget -q --spider http://localhost:${PORT:-9090}/health || exit 1'

# Start command
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-9090}"]
