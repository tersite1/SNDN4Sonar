#!/bin/bash
# SR4IR 학습 재개 (NFS 문제 해결)

cd /mnt/server16_hard1/LIGNEX1/SR4IR

# NFS 임시 파일 문제 해결: /tmp (local SSD) 사용
export TMPDIR=/tmp/sr4ir_train_$$
mkdir -p $TMPDIR

# GPU 메모리 클리어 (OOM 방지)
echo "Clearing GPU cache..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# GPU 5번만 사용 (1 GPU, batch_size=2, accumulation=4로 effective batch=8)
echo "Starting training on GPU 5 (1 GPU, batch_size=2, accumulation=4)..."
CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 \
    src/main.py -opt options/det/train_srdn_sonar.yml

# 학습 종료 후 임시 디렉토리 정리
rm -rf $TMPDIR
