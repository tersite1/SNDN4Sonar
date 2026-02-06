#!/bin/bash
# SR4IR 환경 설치 스크립트

echo "==================================="
echo "SR4IR Conda Environment Setup"
echo "==================================="

# Conda 환경 생성
conda create -n sr4ir python=3.8 -y

# 환경 활성화
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sr4ir

# PyTorch 설치 (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 필수 패키지 설치
pip install numpy==1.24.3
pip install scipy
pip install opencv-python
pip install pillow
pip install pyyaml
pip install tqdm
pip install tensorboard
pip install lpips
pip install scikit-image

# Detection 관련
pip install pycocotools

# Wandb (optional, Go 필요 없는 버전)
pip install wandb==0.16.6

echo "==================================="
echo "설치 완료!"
echo "사용법: conda activate sr4ir"
echo "==================================="
