#!/bin/bash
# SonarSR 모든 경우 평가 (HR, noisy_bilinear, SR)

cd /mnt/server16_hard1/LIGNEX1/SR4IR

DETECTOR_PATH=/mnt/server16_hard1/LIGNEX1/SR4IR/runs_det_gray1ch/combined_1ch/best_map.pth
LABEL_DIR=/mnt/server16_hard1/LIGNEX1/yolo_data/combined/test/labels
BASE_DIR=/mnt/server16_hard0/kangwook/LIGNex1/SonarSR/eval/test_focus

echo "========================================================================"
echo "Evaluating HR (Ground Truth High Resolution)"
echo "========================================================================"
CUDA_VISIBLE_DEVICES=0 python eval_sonarsr_patches.py \
    --sr_dir ${BASE_DIR}/hr \
    --label_dir ${LABEL_DIR} \
    --detector_path ${DETECTOR_PATH} \
    --num_classes 3 \
    --device cuda:0 \
    --conf_threshold 0.001 \
    --suffix ""

echo ""
echo "========================================================================"
echo "Evaluating Noisy Bilinear (Baseline)"
echo "========================================================================"
CUDA_VISIBLE_DEVICES=0 python eval_sonarsr_patches.py \
    --sr_dir ${BASE_DIR}/noisy_bilinear_0.2_0.5 \
    --label_dir ${LABEL_DIR} \
    --detector_path ${DETECTOR_PATH} \
    --num_classes 3 \
    --device cuda:0 \
    --conf_threshold 0.001 \
    --suffix ""

echo ""
echo "========================================================================"
echo "Evaluating SonarSR (Super-Resolution)"
echo "========================================================================"
CUDA_VISIBLE_DEVICES=0 python eval_sonarsr_patches.py \
    --sr_dir ${BASE_DIR}/sr_0.2_0.5_windowattn_BLresidual \
    --label_dir ${LABEL_DIR} \
    --detector_path ${DETECTOR_PATH} \
    --num_classes 3 \
    --device cuda:0 \
    --conf_threshold 0.001 \
    --suffix "_sr"

echo ""
echo "========================================================================"
echo "All evaluations completed!"
echo "========================================================================"
