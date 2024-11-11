FROM ubuntu:22.04

WORKDIR /content

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV PATH="/home/camenduru/.local/bin:/usr/local/cuda/bin:${PATH}"

RUN apt update -y && apt install -y software-properties-common build-essential \
    libgl1 libglib2.0-0 zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev && \
    add-apt-repository -y ppa:git-core/ppa && apt update -y && \
    apt install -y python-is-python3 python3-pip sudo nano aria2 curl wget git git-lfs unzip unrar ffmpeg && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run -d /content -o cuda_12.6.2_560.35.03_linux.run && sh cuda_12.6.2_560.35.03_linux.run --silent --toolkit && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf && ldconfig && \
    git clone https://github.com/aristocratos/btop /content/btop && cd /content/btop && make && make install && \
    adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home
    
USER camenduru

RUN pip install torch==2.5.0+cu124 torchvision==0.20.0+cu124 torchaudio==2.5.0+cu124 torchtext==0.18.0 torchdata==0.8.0 --extra-index-url https://download.pytorch.org/whl/cu124 && \
    pip install xformers==0.0.28.post2 https://github.com/camenduru/wheels/releases/download/torch-2.5.0-cu124/flash_attn-2.6.3-cp310-cp310-linux_x86_64.whl && \
    pip install opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod && \
    pip install torchsde einops diffusers transformers accelerate peft timm && \
    git clone https://github.com/comfyanonymous/ComfyUI /content/ComfyUI && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager /content/ComfyUI/custom_nodes/ComfyUI-Manager && \
    git clone https://github.com/smthemex/ComfyUI_InstantIR_Wrapper /content/ComfyUI/custom_nodes/ComfyUI_InstantIR_Wrapper && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/InstantX/InstantIR/resolve/main/models/adapter.pt -d /content/ComfyUI/models/InstantIR/models -o adapter.pt  && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/InstantX/InstantIR/resolve/main/models/aggregator.pt -d /content/ComfyUI/models/InstantIR/models -o aggregator.pt  && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/InstantX/InstantIR/resolve/main/models/previewer_lora_weights.bin -d /content/ComfyUI/models/InstantIR/models -o previewer_lora_weights.bin  && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/facebook/dinov2-large/raw/main/config.json -d /content/ComfyUI/models/dinov2 -o config.json  && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/facebook/dinov2-large/resolve/main/model.safetensors -d /content/ComfyUI/models/dinov2 -o model.safetensors  && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/facebook/dinov2-large/raw/main/preprocessor_config.json -d /content/ComfyUI/models/dinov2 -o preprocessor_config.json  && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors -d /content/ComfyUI/models/loras/sdxl/lcm -o pytorch_lora_weights.safetensors  && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/ultralytics/resolve/main/dreamshaperXL_lightningDPMSDE.safetensors -d /content/ComfyUI/models/checkpoints -o dreamshaperXL_lightningDPMSDE.safetensors

COPY ./worker_runpod.py /content/ComfyUI/worker_runpod.py
WORKDIR /content/ComfyUI
CMD python worker_runpod.py