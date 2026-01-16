#!/bin/bash

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.6.8/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl

# Download base checkpoint
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B

echo "Installation complete"