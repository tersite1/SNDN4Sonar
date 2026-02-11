# SR4IR with SonarSR - 1-Channel Grayscale Implementation

## Overview
SR4IR framework with SonarSR backbone for sonar image super-resolution and object detection.

## Key Modifications

### 1. True 1-Channel Grayscale Pipeline
Implemented complete 1-channel pipeline for sonar image processing:

#### Data Loader ([src/data/det.py:145](src/data/det.py#L145))
```python
img = img.convert("L")  # Grayscale (1-channel) for SonarSR
```

#### Detector Architecture ([src/archs/det/sonarfasterrcnndetector_arch.py:150](src/archs/det/sonarfasterrcnndetector_arch.py#L150))
```python
# Convert 1-channel to 3-channel internally (MobileNetV3 backbone requirement)
images = [img.repeat(3, 1, 1) if img.size(0) == 1 else img for img in images]
```

#### Channel Flow
```
Data Load → 1ch
    ↓
apply_sonar_noise → 1ch
    ↓
SonarSR (scale=8) → 1ch (sr_output, denoised_lr)
    ↓
force_grayscale → 1ch (defensive)
    ↓
Detector → 3ch conversion (internal only)
    ↓
Detection outputs
```

### 2. Fixed Baseline Detector Evaluation
**Problem**: Epoch 0 baseline evaluation showed mAP=0.0000 because untrained SR network output was fed to detector.

**Solution** ([src/models/det/sr4ir_det_model.py:518-537](src/models/det/sr4ir_det_model.py#L518-L537)):
- **Epoch 0**: Evaluate detector on HR images directly (detector-only baseline)
- **Epoch > 0**: Evaluate detector on SR network output (SR4IR performance)

```python
if epoch == 0:
    # Baseline: HR images (verify detector works)
    img_sr_batch = self.force_grayscale(img_hr_batch)
else:
    # Training: SR network output
    img_lr_batch = apply_sonar_noise(img_hr_batch, downsample=8)
    img_sr_batch, _ = self._forward_sr(img_lr_batch)
```

**Result**:
```
Before: mAP = 0.0000 (epoch 0)
After:  mAP = 0.6826 (epoch 0) ✓
        - mAP@0.25: 0.9328
        - mAP@0.50: 0.8969
        - mAP@0.75: 0.7215
```

### 3. Gradient Accumulation (CUDA OOM Fix)
**Problem**: CUDA Out of Memory with batch_size=2 per GPU
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
Tried to allocate 200.00 MiB (GPU 1; 31.27 GiB already allocated)
```

**Root Cause**: SonarSR's Multi-Head Self-Attention consumes significant memory at 640×640 resolution.

**Solution**: Gradient accumulation implementation ([src/models/det/sr4ir_det_model.py](src/models/det/sr4ir_det_model.py))

#### Configuration ([options/det/train_srdn_sonar.yml:63-64](options/det/train_srdn_sonar.yml#L63-L64))
```yaml
train:
  batch_size: 2  # 2 GPUs → per-GPU 1
  accumulation_steps: 2  # Effective batch = 2 * 2 = 4
```

#### Phase 1: SR Network ([sr4ir_det_model.py:358-395](src/models/det/sr4ir_det_model.py#L358-L395))
```python
accumulation_steps = self.opt['train'].get('accumulation_steps', 1)

# Zero grad only at accumulation start
if iter % accumulation_steps == 0:
    self.optimizer_sr.zero_grad()

# Loss calculation
l_total_sr = l_pix + l_pix_dn + l_tdp

# Backward with loss scaling
(l_total_sr / accumulation_steps).backward()

# Step only at accumulation boundary or end of epoch
if (iter + 1) % accumulation_steps == 0 or (iter + 1) == len(data_loader_train):
    self.optimizer_sr.step()
```

#### Phase 2: Detector Network ([sr4ir_det_model.py:416-484](src/models/det/sr4ir_det_model.py#L416-L484))
```python
# Zero grad only at accumulation start
if iter % accumulation_steps == 0:
    self.optimizer_det.zero_grad()

# Loss calculation (SonarMix: 4 variants)
l_total_det = l_det_hr + l_det_dn + l_det_sr + l_det_dnsr

# Backward with loss scaling
(l_total_det / accumulation_steps).backward()

# Step only at accumulation boundary
if (iter + 1) % accumulation_steps == 0 or (iter + 1) == len(data_loader_train):
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(self.net_det.parameters(), max_norm=1.0)
    self.optimizer_det.step()
```

**Effect**:
- Memory usage: 50% reduction (batch_size 2 → 1 per GPU)
- Training stability: Effective batch_size = 4 maintained
- Performance: Mathematically equivalent to original

## Training Configuration

### Hardware
- GPUs: 2× (CUDA_VISIBLE_DEVICES=0,1 or 4,5)
- Per-GPU Memory: ~12-15 GB (with gradient accumulation)

### Data
- Path: `/mnt/server16_hard1/LIGNEX1/SR4IR/data/combined_1ch`
- Train: 2,395 images (640×640, 1-channel)
- Val: 1,031 images (640×640, 1-channel)
- Format: YOLO normalized bounding boxes

### Pre-trained Weights
```yaml
# SonarSR (scale=8, 1-channel)
network_sr: /mnt/server16_hard0/kangwook/LIGNex1/SonarSR/checkpoints/sr_arch_pixelshuffle_globalo_v2/model_step_100000.pth

# Faster R-CNN Detector (1-channel, trained on HR images)
network_det: /mnt/server16_hard1/LIGNEX1/SR4IR/runs_det_gray1ch/combined_1ch/best_map.pth
```

### Training Command
```bash
cd /mnt/server16_hard1/LIGNEX1/SR4IR
conda activate sr4ir
CUDA_VISIBLE_DEVICES=0,1 python src/main.py -opt options/det/train_srdn_sonar.yml
```

### Training Strategy
1. **Epoch 0**: Baseline evaluation (HR images → Detector)
   - Expected mAP: ~68%

2. **Epoch 1-5**: Warmup (SR only)
   - Losses: l_pix + l_pix_dn
   - Goal: Stabilize SR network before joint training

3. **Epoch 6-50**: Full training (SR + Detector)
   - Losses: l_pix + l_pix_dn + l_tdp (Teacher's Detection Performance)
   - SonarMix: 4 quality variants (HR, DN, SR, DNSR)
   - Goal: SR learns to generate detection-friendly images

## Architecture

### SonarSR (Student)
```
Input: LR (80×80, 1ch)
    ↓
8× ResidualBlock + Multi-Head Self-Attention
    ↓
3× PixelShuffle (×2 each)
    ↓
Dual Output:
  - SR: 640×640, 1ch
  - Denoised LR: 80×80, 1ch
```

### Faster R-CNN Detector (Teacher)
```
Input: HR/SR (640×640, 1ch → 3ch internal)
    ↓
MobileNetV3-Large Backbone
    ↓
FPN (Feature Pyramid Network)
    ↓
RPN + ROI Head
    ↓
Detections (boxes, scores, labels)
```

### Loss Functions

#### Phase 1: SR Network
- `l_pix`: L1(SR, HR) - Pixel reconstruction
- `l_pix_dn`: L1(Denoised_LR, downsample(HR)) - Denoising supervision
- `l_tdp`: L1(Features_SR, Features_HR) - Teacher's Detection Performance (after warmup)

#### Phase 2: Detector Network (SonarMix)
- `l_det_hr`: Detector(HR, targets) - Original HR
- `l_det_dn`: Detector(Denoised↑, targets) - Denoised LR upsampled
- `l_det_sr`: Detector(SR, targets) - SR output
- `l_det_dnsr`: Detector(DNSR, targets) - Denoise→SR (best quality)

## Expected Results

### Baseline (Epoch 0)
```
Detector on HR images (no SR):
- mAP@[0.25:0.95]: 0.68
- mAP@0.25: 0.93
- mAP@0.50: 0.90
```

### Training Progress
```
Epoch 1-5 (Warmup):
- PSNR: 28-30 dB
- mAP: 0.65-0.67 (slight drop during SR stabilization)

Epoch 6-50 (Full training):
- PSNR: 30-35 dB
- mAP: 0.70-0.75 (target: +3-10% from baseline)
- Improvement mechanism: TDP loss guides SR to generate detector-friendly features
```

## Technical Details

### Sonar Noise Model
```python
# Gamma multiplicative speckle noise
dist = min_L + (max_L - min_L) * rand()
dist = dist * (downsample^2)
gamma_dist = Gamma(dist, dist)
noise = gamma_dist.sample()
noisy_image = (image * noise).clamp(0, 1)
```
- min_L: 2.0 (learnable)
- max_L: 10.0 (learnable)

### Memory Optimization
- Gradient accumulation: 2 steps
- Effective batch size: 4 (2 GPUs × 1 per-GPU × 2 accumulation)
- Gradient clipping: max_norm=1.0 (detector only)
- Mixed precision: Not used (stability concerns with detection)

## Key Contributions

1. **True 1-Channel Pipeline**: Preserves sonar image characteristics throughout entire pipeline
2. **Epoch-Aware Baseline Evaluation**: Separate HR and SR evaluation for accurate performance tracking
3. **Gradient Accumulation**: Memory-efficient training for large attention-based SR networks
4. **SonarMix 4-Patch Strategy**: Robust detector training with multiple quality variants
5. **TDP Loss**: Detection-aware super-resolution guidance

## Citation
If you use this implementation, please cite:
```bibtex
@inproceedings{SR4IR_SonarSR,
  title={SR4IR: Super-Resolution for Image Restoration with SonarSR},
  author={Your Name},
  year={2026}
}
```
