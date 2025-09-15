FROM ubuntu:24.04
ARG CUDA_VERSION=11.8.0
ENV CUDA_VERSION=${CUDA_VERSION}
ENV DEBIAN_FRONTEND=noninteractive
# 基础系统更新和工具安装
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated ca-certificates \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    wget git \
    curl \
    build-essential \
    gcc-11 g++-11 \
    libgl1-mesa-dev \
    libglib2.0-0 \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*
# 安装 Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*
# 安装 pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py
# 设置 python/pip 默认
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 100 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 100 && \
    update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.11 100
# 设置 GCC 为 11（兼容 nvcc）
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100; \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100
# CUDA 架构变量
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
# CUDA toolkit
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install -y \
    cuda-nvcc-11-8 \
    cuda-cudart-dev-11-8 \
    cuda-driver-dev-11-8 \
    libcublas-dev-11-8 \
    libcufft-dev-11-8 \
    libcurand-dev-11-8 \
    libcusolver-dev-11-8 \
    libcusparse-dev-11-8 \
    && rm -f cuda-keyring_1.0-1_all.deb && \
    rm -rf /var/lib/apt/lists/*
# CUDA 环境变量
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/usr/local/lib
# cmake & ninja
RUN apt-get update && \
    apt-get install -y cmake ninja-build && \
    rm -rf /var/lib/apt/lists/*
# 创建虚拟环境
RUN python3.11 -m venv /opt/3dgrut
ENV PATH=/opt/3dgrut/bin:$PATH
WORKDIR /workspace
COPY . .
# 升级 pip
RUN pip install --upgrade pip
# 安装 PyTorch（必须在 requirements.txt 前）
RUN if [ "$CUDA_VERSION" = "11.8.0" ]; then \
        pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
            --index-url https://download.pytorch.org/whl/cu118 && \
        pip install "numpy<2.0"; \
    elif [ "$CUDA_VERSION" = "12.8.1" ]; then \
        pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu128 && \
        pip install --force-reinstall "numpy<2"; \
    else \
        echo "Unsupported CUDA version: $CUDA_VERSION"; \
        exit 1; \
    fi
# 安装 OpenGL
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    mesa-common-dev \
    libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*
# 安装 Kaolin（必须在 requirements.txt 前）
RUN if [ "$CUDA_VERSION" = "11.8.0" ]; then \
        pip install --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.2_cu118.html kaolin==0.17.0; \
    elif [ "$CUDA_VERSION" = "12.8.1" ]; then \
        rm -fr thirdparty/kaolin && \
        git clone --recursive https://github.com/NVIDIAGameWorks/kaolin.git thirdparty/kaolin && \
        cd thirdparty/kaolin && \
        git checkout c2da967b9e0d8e3ebdbd65d3e8464d7e39005203 && \
        sed -i 's!AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats_in.type()!AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats_in.scalar_type()!g' kaolin/csrc/render/spc/raytrace_cuda.cu && \
        pip install --upgrade pip && \
        pip install --no-cache-dir ninja imageio imageio-ffmpeg && \
        pip install --no-cache-dir \
            -r tools/viz_requirements.txt \
            -r tools/requirements.txt \
            -r tools/build_requirements.txt && \
        IGNORE_TORCH_VER=1 python setup.py install && \
        cd ../.. && rm -fr thirdparty/kaolin; \
    fi
# 初始化子模块
RUN git submodule update --init --recursive

RUN pip install --upgrade pip setuptools wheel packaging

# 安装 requirements（现在 torch 已存在，不会报错）
RUN CUDA_HOME=/usr/local/cuda-11.8 \
    PATH=/usr/local/cuda-11.8/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH \
    pip install --no-build-isolation -r requirements.txt
# 开发模式安装项目
RUN pip install -e .
# 验证安装
RUN python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
# 自动激活环境
RUN echo 'source /opt/3dgrut/bin/activate' >> ~/.bashrc
RUN echo 'echo \"3DGRUT environment activated - Python $(python --version)\"' >> ~/.bashrc
CMD ["/bin/bash"]