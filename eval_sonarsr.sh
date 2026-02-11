#!/bin/bash
# SonarSR 패치들 평가

cd /mnt/server16_hard1/LIGNEX1/SR4IR

CUDA_VISIBLE_DEVICES=0 python eval_sonarsr_patches.py \
    --sr_dir /mnt/server16_hard0/kangwook/LIGNex1/SonarSR/eval/test_focus/sr_0.2_0.5_windowattn_BLresidual \
    --label_dir /mnt/server16_hard1/LIGNEX1/yolo_data/combined/test/labels \
    --detector_path /mnt/server16_hard1/LIGNEX1/SR4IR/runs_det_gray1ch/combined_1ch/best_map.pth \
    --num_classes 3 \
    --device cuda:0 \
    --conf_threshold 0.001
