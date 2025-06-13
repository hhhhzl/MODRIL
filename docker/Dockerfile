# docker/Dockerfile
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /workspace
COPY . /workspace

RUN chmod +x scripts/copy_gym_robotics.sh && \
    scripts/install_mujoco_linux.sh.sh && \
    scripts/install_mujoco_py_linux.sh


RUN chmod +x scripts/setup.sh && \
    scripts/setup.sh

RUN chmod +x scripts/copy_gym_robotics.sh && \
    scripts/copy_gym_robotics.sh

RUN chmod +x scripts/download_demos.sh && \
    scripts/download_demos.sh

CMD ["bash"]