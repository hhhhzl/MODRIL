# docker/Dockerfile
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      wget gnupg2 ca-certificates lsb-release \
      software-properties-common \
      build-essential git curl vim \
      python3.10 python3.10-dev python3-pip \
      libgl1-mesa-glx libosmesa6-dev libglew-dev \
      xvfb ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 2. NVIDIA CUDA 11.8
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    rm cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      cuda-toolkit-11-8 \
      libcudnn8=8.*-1+cuda11.8 \
      libcudnn8-dev=8.*-1+cuda11.8 && \
    rm -rf /var/lib/apt/lists/*

# 3. CUDA Env
ENV PATH=/usr/local/cuda-11.8/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}

# 4. python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python3 -m pip install --upgrade pip

RUN pip install --no-cache-dir \
      torch==2.1.0+cu118 \
      torchvision==0.16.0+cu118 \
      torchaudio==2.1.0 \
      --extra-index-url https://download.pytorch.org/whl/cu118

WORKDIR /workspace
COPY . /workspace

RUN chmod +x /scripts/install_mujoco_*.sh && \
    /scripts/install_mujoco_linux.sh && \
    /scripts/install_mujoco_py_linux.sh

ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
ENV LD_LIBRARY_PATH=$MUJOCO_PY_MUJOCO_PATH/bin:$LD_LIBRARY_PATH

RUN chmod +x scripts/setup.sh && \
    scripts/setup.sh

RUN chmod +x scripts/copy_gym_robotics.sh && \
    scripts/copy_gym_robotics.sh

RUN chmod +x scripts/download_demos.sh && \
    scripts/download_demos.sh

CMD ["bash"]