# # Stage 1: Builder/Compiler
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CPATH=/usr/local/cuda/include:${CPATH}

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0 \
    &&rm -rf /var/lib/apt/lists/*
    
# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch and torchvision
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install any python packages you need
# COPY requirements.txt requirements.txt
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
ENV FORCE_CUDA 1
ENV TORCH_CUDA_ARCH_LIST "3.5 5.2 6.0 6.1 7.0+PTX"
RUN pip3 install --no-cache-dir --no-deps torch-points-kernels torch-points3d
RUN pip3 install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# # Stage 2: Runtime
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CPATH=/usr/local/cuda/include:${CPATH}
ENV FORCE_CUDA 1
ENV TORCH_CUDA_ARCH_LIST "3.5 5.2 6.0 6.1 7.0+PTX"

RUN apt update && \
    apt install --no-install-recommends -y build-essential software-properties-common git && \
    apt install --no-install-recommends -y \
        python3.10 python3-distutils python3-opencv\
        libx11-6 libgl1-mesa-glx && \
    apt clean && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Set the working directory
WORKDIR /app
# Clone GitHub repository
RUN git clone https://github.com/TuyetHan/CamLidCalib_Trans.git

# Set the entrypoint
#ENTRYPOINT ["python", "train.py"]
ENTRYPOINT [ "/bin/bash" ]



