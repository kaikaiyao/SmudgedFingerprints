# Base image with CUDA 12.6 and MIG tools
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Set timezone non-interactively
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/UTC /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Install system dependencies with Python and MIG support
RUN apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y \
    git \
    cmake \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    gcc-12 \
    g++-12 \
    curl \
    python3.10 \
    python3-pip \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && rm -rf /var/lib/apt/lists/*

# Configure CUDA paths and MIG capabilities
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_VERSION=12.6
ENV CUDA_HOME=/usr/local/cuda
ENV NVIDIA_DISABLE_REQUIRE=1
ENV NVIDIA_VISIBLE_DEVICES=all

# Install PyTorch 2.6.0 with CUDA 12.6 compatibility
RUN pip3 install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0+cu126 \
    torchaudio==2.6.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126

# Install core scientific computing packages
RUN pip install --no-cache-dir \
    numpy>=1.21.0 \
    scipy>=1.7.0 \
    matplotlib>=3.5.0 \
    seaborn>=0.11.0 \
    pandas>=1.3.0 \
    scikit-learn>=1.0.0 \
    scikit-image>=0.18.0 \
    tqdm>=4.60.0

# Install computer vision and image processing packages
RUN pip install --no-cache-dir \
    Pillow>=8.3.0 \
    opencv-python>=4.5.0

# Install deep learning and ML packages
RUN pip install --no-cache-dir \
    torchmetrics \
    torch-fidelity \
    lpips

# Install HuggingFace ecosystem packages
RUN pip install --no-cache-dir \
    diffusers>=0.21.0 \
    transformers>=4.20.0 \
    accelerate>=0.20.0 \
    safetensors>=0.3.0 \
    datasets>=2.0.0 \
    huggingface_hub>=0.16.0 \
    tokenizers>=0.13.0

# Install configuration and utility packages  
RUN pip install --no-cache-dir \
    pyyaml>=6.0 \
    click>=8.0.0 \
    requests>=2.25.0 \
    urllib3>=1.26.0

# Install additional packages for specific functionality
RUN pip install --no-cache-dir \
    ninja>=1.10.2 \
    cryptography>=3.4.0 \
    psutil>=5.8.0 \
    packaging>=21.0 \
    filelock>=3.8.0 \
    regex>=2022.1.18 \
    tabulate \
    PyTurboJPEG

RUN apt-get update && \
    apt-get install -y libturbojpeg

# Configure writable directories for PyTorch extensions
ENV TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
RUN mkdir -p ${TORCH_EXTENSIONS_DIR} && chmod -R 777 ${TORCH_EXTENSIONS_DIR}

ARG CACHEBUST=1

CMD ["true"]