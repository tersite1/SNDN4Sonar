import os
import os.path as osp
import torch
import numpy as np
import torch.nn.functional as F

from archs import build_network
from losses import build_loss
from torch.nn.functional import interpolate
from utils.common import save_on_master, quantize, calculate_psnr_batch, visualize_image, calculate_lpips_batch, is_main_process
from utils.det import MetricLogger, SmoothedValue, get_coco_api_from_dataset, _get_iou_types, CocoEvaluator

from .base_model import BaseModel

def apply_sonar_noise(image, downsample=4, min_L=2.0, max_L=10.0, use_rsample=False):
    # 1. 다운샘플링 (LR 입력 생성)
    if downsample != 1:
        downsample_image = interpolate(image, scale_factor=1 / downsample, mode='bilinear', align_corners=False)
    else:
        downsample_image = image
    # 2. 노이즈 파라미터 생성
    if torch.is_tensor(min_L):
        min_L = min_L.to(device=image.device)
    else:
        min_L = torch.tensor(float(min_L), device=image.device)
    if torch.is_tensor(max_L):
        max_L = max_L.to(device=image.device)
    else:
        max_L = torch.tensor(float(max_L), device=image.device)
    dist = min_L + (max_L - min_L) * torch.rand((), device=image.device)
    dist = dist * (downsample**2)
    gamma_dist = torch.distributions.Gamma(dist, dist)
    
    # 3. 노이즈 생성
    if use_rsample and gamma_dist.has_rsample:
        noise = gamma_dist.rsample(downsample_image.shape)
    else:
        noise = gamma_dist.sample(downsample_image.shape)
    noise = noise.to(device=image.device)
    
    # 4. 스펙클 노이즈 적용 (0~1 범위 가정: 단순히 곱하기)
    # 소나 노이즈: Signal * Noise
    noisy_image = (downsample_image * noise).clamp(0, 1)
    
    return noisy_image

def make_model(opt):
    return SR4IRDetectionModel(opt)


class SR4IRDetectionModel(BaseModel):
    """Base Super-Resolution model for Object Detection."""

    def __init__(self, opt):
        super().__init__(opt)

        self.learnable_noise = self.opt.get('train', {}).get('learnable_noise', False)
        if self.learnable_noise:
            init_min = float(self.opt['train'].get('noise_min_L', 2.0))
            init_max = float(self.opt['train'].get('noise_max_L', 10.0))
            if init_max <= init_min:
                init_max = init_min + 1.0
            self.noise_L_min = torch.nn.Parameter(torch.tensor(init_min, device=self.device))
            self.noise_L_delta = torch.nn.Parameter(torch.tensor(init_max - init_min, device=self.device))

        # define network up
        self.net_up = self.model_to_device(torch.nn.UpsamplingBilinear2d(scale_factor=self.scale), is_trainable=False)

        # define network sr
        opt['network_sr']['scale'] = self.scale
        self.net_sr = build_network(opt['network_sr'], self.text_logger, tag='net_sr')
        self.load_network(self.net_sr, name='network_sr', tag='net_sr')
        self.net_sr = self.model_to_device(self.net_sr, is_trainable=True)
        self.print_network(self.net_sr, tag='net_sr')

        # Check if using SonarSR (returns dual output)
        self.use_sonarsr = opt['network_sr'].get('name', '').lower() == 'sonarsr'
        
        # define network detction
        self.net_det = build_network(opt['network_det'], self.text_logger, task=self.task, tag='net_det')
        self.load_network(self.net_det, name='network_det', tag='net_det')
        self.net_det = self.model_to_device(self.net_det, is_trainable=True)
        self.print_network(self.net_det, tag='net_det')
        
    def set_mode(self, mode):
        if mode == 'train':
            self.net_sr.train()
            self.net_det.train()
        elif mode == 'eval':
            self.net_sr.eval()
            self.net_det.eval()
        else:
            raise NotImplementedError(f"mode {mode} is not supported")

    def get_noise_L_range(self):
        if not self.learnable_noise:
            min_L = self.opt.get('train', {}).get('noise_min_L', 2.0)
            max_L = self.opt.get('train', {}).get('noise_max_L', 10.0)
            return min_L, max_L
        min_L = F.softplus(self.noise_L_min)
        max_L = min_L + F.softplus(self.noise_L_delta)
        return min_L, max_L
    def force_grayscale(self, img_batch):
        """
        Keep grayscale (1-channel) format - detector handles channel conversion internally.
        Uses Luma conversion (same as Pillow): 0.299*R + 0.587*G + 0.114*B
        """
        if img_batch.dim() == 4:
            if img_batch.size(1) == 3:
                # 3채널 → grayscale (1채널) using Luma weights
                luma = 0.299 * img_batch[:, 0:1, :, :] + \
                       0.587 * img_batch[:, 1:2, :, :] + \
                       0.114 * img_batch[:, 2:3, :, :]
                return luma
        return img_batch

    def _forward_sr(self, img_lr_batch):
        """
        Forward SR network with support for both single and dual output models.

        For SonarSR: returns (sr_output, denoised_lr)
        For SwinIR/others: returns (sr_output, None)
        """
        output = self.net_sr(img_lr_batch)
        if self.use_sonarsr:
            # SonarSR returns (final_x, denoised_x)
            img_sr_batch, img_denoised_batch = output
            return img_sr_batch, img_denoised_batch
        else:
            # Standard SR models return single output
            return output, None


    #sonar SR 에서 가져옴
    def maybe_save_sr(self, img_sr_batch, filename=None, prob=0.01, suffix=None):
        if prob <= 0:
            return
        if torch.rand(1).item() >= prob:
            return
        save_dir = osp.join(self.exp_dir, 'sr_samples')
        os.makedirs(save_dir, exist_ok=True)
        if img_sr_batch.dim() == 4:
            img = img_sr_batch[0]
        else:
            img = img_sr_batch
        if filename is None:
            filename = 'sr_sample.png'
        visualize_image(img, save_dir, filename, img_range=1.0, suffix=suffix)

    def _log_coco_metrics(self, coco_evaluator, epoch, prefix="det", step=None):
        if coco_evaluator is None:
            return
        stat_names = [
            "ap",
            "ap50",
            "ap75",
            "ap_small",
            "ap_medium",
            "ap_large",
            "ar1",
            "ar10",
            "ar100",
            "ar_small",
            "ar_medium",
            "ar_large",
        ]

        # Track best AP for checkpoint saving
        best_ap = None

        for iou_type, coco_eval in coco_evaluator.coco_eval.items():
            stats = getattr(coco_eval, "stats", None)
            if stats is None:
                continue
            metrics = {}
            for idx, name in enumerate(stat_names):
                if idx >= len(stats):
                    break
                key = f"{prefix}/{iou_type}/{name}"
                value = float(stats[idx])
                metrics[key] = value
                if self.is_train and hasattr(self, "tb_logger"):
                    self.tb_logger.add_scalar(key, value, epoch)

            # Add summary metrics for main detection performance
            if len(stats) >= 3:
                # Extract standard COCO metrics
                # NOTE: stats[0] is now average over [0.25, 0.5, ..., 0.95] due to modified iouThrs
                ap = float(stats[0])  # mAP @ IoU=0.25:0.95 (11 thresholds)
                ap50 = float(stats[1])  # mAP @ IoU=0.50 (index 1 in our iouThrs array)
                ap75 = float(stats[2])  # mAP @ IoU=0.75 (index 6 in our iouThrs array)

                # Extract mAP@0.25 from precision array
                # precision shape: [T, R, K, A, M] where T=IoU thresholds, R=recall, K=class, A=area, M=maxDets
                coco_eval = coco_evaluator.coco_eval[iou_type]
                if hasattr(coco_eval, 'eval') and coco_eval.eval is not None:
                    # precision[0, :, :, 0, 2] → IoU@0.25, all areas, maxDets=100
                    p25 = coco_eval.eval['precision'][0, :, :, 0, 2]
                    # Compute mean AP@0.25 (average over valid recall points and classes)
                    ap25 = float(np.mean(p25[p25 > -1])) if p25[p25 > -1].size > 0 else 0.0
                else:
                    ap25 = 0.0

                # Track best AP (using standard mAP 0.5:0.95 for compatibility)
                if best_ap is None or ap > best_ap:
                    best_ap = ap

                # Add clear summary metrics
                summary_key = f"{prefix}/{iou_type}"
                metrics[f"{summary_key}/mAP"] = ap
                metrics[f"{summary_key}/mAP25"] = ap25
                metrics[f"{summary_key}/mAP50"] = ap50
                metrics[f"{summary_key}/mAP75"] = ap75

                # Log to console for quick checking
                self.text_logger.write(
                    f"[Detection Performance] mAP: {ap:.4f} | mAP25: {ap25:.4f} | mAP50: {ap50:.4f} | mAP75: {ap75:.4f}"
                )

                # Save detector metrics to file
                if hasattr(self, 'detector_metrics_file'):
                    with open(self.detector_metrics_file, 'a') as f:
                        f.write(f"{epoch},{ap:.6f},{ap25:.6f},{ap50:.6f},{ap75:.6f}\n")

            if metrics:
                self.wandb_log(metrics, step=step if step is not None else epoch)

        return best_ap

        
    def init_training_settings(self, data_loader_train):
        self.set_mode(mode='train')
        train_opt = self.opt['train']

        # phase 1
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt'], self.text_logger).to(self.device)

        if train_opt.get('pixel_dn_opt'):
            # Denoised LR pixel loss (for SonarSR intermediate supervision)
            self.cri_pix_dn = build_loss(train_opt['pixel_dn_opt'], self.text_logger).to(self.device)

        if train_opt.get('tdp_opt'):
            # task driven perceptual loss
            self.cri_tdp = build_loss(train_opt['tdp_opt'], self.text_logger).to(self.device)

        # phase 2 - SonarMix 4-Patch Strategy
        if train_opt.get('det_hr_opt'):
            self.cri_det_hr = build_loss(train_opt['det_hr_opt'], self.text_logger).to(self.device)

        if train_opt.get('det_dn_opt'):
            self.cri_det_dn = build_loss(train_opt['det_dn_opt'], self.text_logger).to(self.device)

        if train_opt.get('det_sr_opt'):
            self.cri_det_sr = build_loss(train_opt['det_sr_opt'], self.text_logger).to(self.device)

        if train_opt.get('det_dnsr_opt'):
            self.cri_det_dnsr = build_loss(train_opt['det_dnsr_opt'], self.text_logger).to(self.device)

        if train_opt.get('det_cqmix_opt'):
            self.cri_det_cqmix = build_loss(train_opt['det_cqmix_opt'], self.text_logger).to(self.device)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers(len(data_loader_train), name='sr', optimizer=self.optimizer_sr)
        self.setup_schedulers(len(data_loader_train), name='det', optimizer=self.optimizer_det)

        # set up saving directories
        os.makedirs(osp.join(self.exp_dir, 'models'), exist_ok=True)
        os.makedirs(osp.join(self.exp_dir, 'checkpoints'), exist_ok=True)

        # Create detector metrics file
        self.detector_metrics_file = osp.join(self.exp_dir, 'detector_metrics.txt')
        if is_main_process():
            with open(self.detector_metrics_file, 'w') as f:
                f.write("epoch,mAP,mAP25,mAP50,mAP75\n")

        # eval freq
        self.eval_freq = train_opt.get('eval_freq', 1)

        # warmup epoch
        self.warmup_epoch = train_opt.get('warmup_epoch', -1)
        self.text_logger.write("NOTICE: total epoch: {}, warmup epoch: {}".format(train_opt['epoch'], self.warmup_epoch))
        
    def setup_optimizers(self):
        train_opt = self.opt['train']
        
        # optimizer sr
        optim_type = train_opt['optim_sr'].pop('type')
        sr_params = list(self.net_sr.parameters())
        if self.learnable_noise:
            sr_params += [self.noise_L_min, self.noise_L_delta]
        self.optimizer_sr = self.get_optimizer(optim_type, sr_params, **train_opt['optim_sr'])
        self.optimizers.append(self.optimizer_sr)
        
        # optimizer det
        optim_type = train_opt['optim_det'].pop('type')
        net_det_parameters = [p for p in self.net_det.parameters() if p.requires_grad]
        self.optimizer_det = self.get_optimizer(optim_type, net_det_parameters, **train_opt['optim_det'])
        self.optimizers.append(self.optimizer_det)
            
    def train_one_epoch(self, data_loader_train, train_sampler, epoch):
        self.set_mode(mode='train')
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr_sr", SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("lr_det", SmoothedValue(window_size=1, fmt="{value}"))
        
        if self.dist:
            train_sampler.set_epoch(epoch)
            
        if epoch < self.warmup_epoch + 1:
            self.text_logger.write("NOTICE: Doing warm-up")
            
        # NOTE: without warmup, training explodes!!
        lr_scheduler_s = None
        lr_scheduler_d = None
        if epoch == 1:
            warmup_factor = 1.0 / len(data_loader_train)
            warmup_iters = len(data_loader_train)
            lr_scheduler_s = torch.optim.lr_scheduler.LinearLR(
                self.optimizer_sr, start_factor=warmup_factor, total_iters=warmup_iters)
            lr_scheduler_d = torch.optim.lr_scheduler.LinearLR(
                self.optimizer_det, start_factor=warmup_factor, total_iters=warmup_iters)

        header = f"Epoch: [{epoch}, Name {self.opt['name']}]"
        for iter, (img_hr_list, target_list) in enumerate(metric_logger.log_every(data_loader_train, self.opt['print_freq'], self.text_logger, header)):
            img_hr_list = list(img_hr.to(self.device) for img_hr in img_hr_list)
            target_list = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in target_list]
            current_iter = iter + len(data_loader_train)*(epoch-1)
            self.current_iter = current_iter

            # make on-the-fly LR image
            img_hr_batch = self.list_to_batch(img_hr_list)

            # 노이즈랑 SR 동시 수행
            min_L, max_L = self.get_noise_L_range()
            img_lr_batch = apply_sonar_noise(
                img_hr_batch,           
                downsample=self.scale,  
                min_L=min_L, max_L=max_L,
                use_rsample=self.learnable_noise,
            )

            # phase 1;
            # update net_sr, freeze net_cls
            img_sr_batch, img_denoised_batch = self._forward_sr(img_lr_batch)
            img_sr_batch = self.force_grayscale(img_sr_batch)
            save_prob = self.opt['train'].get('sr_save_prob', 0.01)
            save_name = f"train_e{epoch:03d}_iter{current_iter:07d}.png"
            self.maybe_save_sr(img_sr_batch, filename=save_name, prob=save_prob)
            img_sr_list = self.batch_to_list(img_sr_batch, img_list=img_hr_list)
            for p in self.net_det.parameters(): p.requires_grad = False
            self.optimizer_sr.zero_grad()
            l_total_sr = 0
            if hasattr(self, 'cri_pix'):
                l_pix = self.cri_pix(img_sr_batch, img_hr_batch)
                metric_logger.meters["l_pix"].update(l_pix.item())
                self.log_scalar('losses/l_pix', l_pix.item(), current_iter)
                l_total_sr += l_pix

            # Denoised LR supervision (SonarSR intermediate output)
            if hasattr(self, 'cri_pix_dn') and img_denoised_batch is not None:
                # Downsample HR to LR size for GT comparison
                img_lr_gt = interpolate(img_hr_batch, scale_factor=1/self.scale, mode='bicubic', align_corners=False)
                l_pix_dn = self.cri_pix_dn(img_denoised_batch, img_lr_gt)
                metric_logger.meters["l_pix_dn"].update(l_pix_dn.item())
                self.log_scalar('losses/l_pix_dn', l_pix_dn.item(), current_iter)
                l_total_sr += l_pix_dn

            if epoch > self.warmup_epoch:
                if hasattr(self, 'cri_tdp'):
                    self.net_det.eval()
                    _, _, feat_sr = self.net_det(img_sr_list, return_feats=True)
                    _, _, feat_hr = self.net_det(img_hr_list, return_feats=True)
                    self.net_det.train()
                    
                    l_tdp = self.cri_tdp(feat_sr['features'], feat_hr['features'])
                    metric_logger.meters["l_tdp"].update(l_tdp.item()) 
                    self.log_scalar('losses/l_tdp', l_tdp.item(), current_iter)
                    l_total_sr += l_tdp
            l_total_sr.backward()
            self.optimizer_sr.step()
            
            # phase 2;
            # update network det, freeze net_cls
            img_sr_batch, img_denoised_batch = self._forward_sr(img_lr_batch)
            img_sr_batch = img_sr_batch.detach()
            img_sr_batch = self.force_grayscale(img_sr_batch)
            img_sr_list = self.batch_to_list(img_sr_batch, img_list=img_hr_list)

            # Prepare denoised variants if using SonarMix
            img_dn_list = None
            img_dnsr_list = None
            if img_denoised_batch is not None:
                # SonarSR provides denoised LR output - upscale to HR resolution
                img_denoised_batch = img_denoised_batch.detach()
                img_denoised_batch = self.force_grayscale(img_denoised_batch)
                # Upscale denoised LR to HR size using bicubic interpolation (keep 1-channel)
                img_dn_batch = interpolate(img_denoised_batch, scale_factor=self.scale, mode='bicubic', align_corners=False)
                img_dn_list = self.batch_to_list(img_dn_batch, img_list=img_hr_list)
                # For denoise+SR variant: img_sr_list already is denoise→SR from SonarSR
                # So we use img_sr_list as dnsr variant
                img_dnsr_list = img_sr_list

            for p in self.net_det.parameters(): p.requires_grad = True
            self.optimizer_det.zero_grad()
            l_total_det = 0
            loss_clip = self.opt['train'].get('det_loss_clip', None)

            # 1. HR original (noisy) - 원본패치
            if hasattr(self, 'cri_det_hr'):
                _, loss_dict_hr = self.net_det(img_hr_list, target_list)
                l_det_hr = self.cri_det_hr(loss_dict_hr)
                if loss_clip and loss_clip > 0:
                    l_det_hr = torch.clamp(l_det_hr, max=loss_clip)
                metric_logger.meters["l_det_hr"].update(l_det_hr.item())
                self.log_scalar('losses/l_det_hr', l_det_hr.item(), current_iter)
                l_total_det += l_det_hr

            # 2. Denoised LR upscaled - 디노이즈 패치
            if hasattr(self, 'cri_det_dn') and img_dn_list is not None:
                _, loss_dict_dn = self.net_det(img_dn_list, target_list)
                l_det_dn = self.cri_det_dn(loss_dict_dn)
                if loss_clip and loss_clip > 0:
                    l_det_dn = torch.clamp(l_det_dn, max=loss_clip)
                metric_logger.meters["l_det_dn"].update(l_det_dn.item())
                self.log_scalar('losses/l_det_dn', l_det_dn.item(), current_iter)
                l_total_det += l_det_dn

            # 3. SR (implicit denoising+SR) - SR패치
            if hasattr(self, 'cri_det_sr'):
                _, loss_dict_sr = self.net_det(img_sr_list, target_list)
                l_det_sr = self.cri_det_sr(loss_dict_sr)
                if loss_clip and loss_clip > 0:
                    l_det_sr = torch.clamp(l_det_sr, max=loss_clip)
                metric_logger.meters["l_det_sr"].update(l_det_sr.item())
                self.log_scalar('losses/l_det_sr', l_det_sr.item(), current_iter)
                l_total_det += l_det_sr

            # 4. Denoise→SR (best quality) - 디노이즈SR패치
            if hasattr(self, 'cri_det_dnsr') and img_dnsr_list is not None:
                _, loss_dict_dnsr = self.net_det(img_dnsr_list, target_list)
                l_det_dnsr = self.cri_det_dnsr(loss_dict_dnsr)
                if loss_clip and loss_clip > 0:
                    l_det_dnsr = torch.clamp(l_det_dnsr, max=loss_clip)
                metric_logger.meters["l_det_dnsr"].update(l_det_dnsr.item())
                self.log_scalar('losses/l_det_dnsr', l_det_dnsr.item(), current_iter)
                l_total_det += l_det_dnsr

            # CQMix (optional, legacy support)
            if hasattr(self, 'cri_det_cqmix'):
                batch_size = len(img_hr_list)
                mask = interpolate((torch.randn(batch_size,1,8,8)).bernoulli_(p=0.5), size=(img_hr_batch.shape[2:]), mode='nearest').to(self.device)
                img_cqmix_batch = img_sr_batch*mask + img_hr_batch*(1-mask)
                img_cqmix_batch = self.force_grayscale(img_cqmix_batch)
                img_cqmix_list = self.batch_to_list(img_cqmix_batch, img_list=img_hr_list)
                _, loss_dict_cqmix = self.net_det(img_cqmix_list, target_list)
                l_det_cqmix = self.cri_det_cqmix(loss_dict_cqmix)
                if loss_clip and loss_clip > 0:
                    l_det_cqmix = torch.clamp(l_det_cqmix, max=loss_clip)
                metric_logger.meters["l_det_cqmix"].update(l_det_cqmix.item())
                self.log_scalar('losses/l_det_cqmix', l_det_cqmix.item(), current_iter)
                l_total_det += l_det_cqmix
            l_total_det.backward()
            max_norm = self.opt['train'].get('det_grad_clip', 1.0)
            if max_norm and max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.net_det.parameters(), max_norm)
            self.optimizer_det.step()
            
            # psnr, lr
            psnr, valid_batch_size = calculate_psnr_batch(quantize(img_sr_batch), img_hr_batch)
            metric_logger.meters["psnr"].update(psnr.item(), n=valid_batch_size)
            metric_logger.update(lr_sr=round(self.optimizer_sr.param_groups[0]["lr"], 8))
            metric_logger.update(lr_det=round(self.optimizer_det.param_groups[0]["lr"], 8))
            
            # update learning rate
            if epoch == 1:
                lr_scheduler_s.step()
                lr_scheduler_d.step()
            else:
                self.update_learning_rate()
        return
            
    @torch.inference_mode()
    def evaluate(self, data_loader_test, epoch=0):
        if hasattr(self, 'eval_freq') and (epoch % self.eval_freq != 0):
            return
        
        self.set_mode(mode='eval')
        metric_logger = MetricLogger(delimiter="  ")
        header = "Test:"
        
        coco = get_coco_api_from_dataset(data_loader_test.dataset)
        iou_types = _get_iou_types(self.net_det)
        coco_evaluator = CocoEvaluator(coco, iou_types)

        # Add IoU@0.25 for more lenient evaluation (sonar detection often needs lower threshold)
        for iou_type in iou_types:
            coco_eval = coco_evaluator.coco_eval[iou_type]
            # Add 0.25 to the standard IoU thresholds [0.5:0.95]
            iouThrs = np.array([0.25, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
            coco_eval.params.iouThrs = iouThrs

        num_processed_samples = 0
        for (img_hr_list, target_list), filename in metric_logger.log_every(data_loader_test, 1000, self.text_logger, header, return_filename=True):
            img_hr_list = list(img_hr.to(self.device) for img_hr in img_hr_list)
            target_list = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in target_list]

            # make on-the-fly LR image
            img_hr_batch = self.list_to_batch(img_hr_list)
            min_L, max_L = self.get_noise_L_range()
            img_lr_batch = apply_sonar_noise(
            img_hr_batch,           
            downsample=self.scale,  
            min_L=min_L, max_L=max_L,
            use_rsample=self.learnable_noise,
            )
            # perform SR
            img_sr_batch, _ = self._forward_sr(img_lr_batch)
            img_sr_batch = self.force_grayscale(img_sr_batch)
            img_sr_list = self.batch_to_list(img_sr_batch, img_list=img_hr_list)

            # object detection
            if torch.cuda.is_available(): torch.cuda.synchronize()
            outputs_sr, _ = self.net_det(img_sr_list)
            outputs_sr = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs_sr]

            # visualizing tool
            if self.opt['test'].get('visualize', False): # and (num_processed_samples < 20):
                self.visualize(img_sr_list[0], outputs_sr[0], filename)
            save_prob = self.opt['test'].get('sr_save_prob', 0.01)
            self.maybe_save_sr(img_sr_batch, filename=filename, prob=save_prob, suffix='sr')

            # evaluation on validation batch
            batch_size = len(img_sr_list)
            psnr, valid_batch_size = calculate_psnr_batch(quantize(img_sr_batch), img_hr_batch)
            metric_logger.meters["psnr"].update(psnr.item(), n=valid_batch_size)
            if self.opt['test'].get('calculate_lpips', False):
                lpips, valid_batch_size = calculate_lpips_batch(quantize(img_sr_batch), img_hr_batch, self.net_lpips)
                metric_logger.meters["lpips"].update(lpips.item(), n=valid_batch_size)
            res = {target["image_id"]: output for target, output in zip(target_list, outputs_sr)}
            coco_evaluator.update(res)
            num_processed_samples += batch_size
    
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        coco_evaluator.synchronize_between_processes()
        
        # logging training state
        metric_summary = f"{header}"
        metric_summary = self.add_metric(metric_summary, 'PSNR', metric_logger.psnr.global_avg, epoch)
        if self.opt['test'].get('calculate_lpips', False):
            metric_summary = self.add_metric(metric_summary, 'LPIPS', metric_logger.lpips.global_avg, epoch)
        self.text_logger.write(metric_summary)
        wandb_step = getattr(self, "current_iter", epoch)
        val_metrics = {"val/psnr": metric_logger.psnr.global_avg}
        if self.opt['test'].get('calculate_lpips', False):
            val_metrics["val/lpips"] = metric_logger.lpips.global_avg
        self.wandb_log(val_metrics, step=wandb_step)
        if self.is_train and hasattr(self, "tb_logger"):
            self.tb_logger.add_scalar("val/psnr", metric_logger.psnr.global_avg, epoch)
            if self.opt['test'].get('calculate_lpips', False):
                self.tb_logger.add_scalar("val/lpips", metric_logger.lpips.global_avg, epoch)

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize(self.text_logger, tag='SR')
        current_ap = self._log_coco_metrics(coco_evaluator, epoch, prefix="val/det", step=wandb_step)

        # Save best checkpoint based on mAP
        if current_ap is not None:
            if not hasattr(self, 'best_ap'):
                self.best_ap = 0.0
            if not hasattr(self, 'baseline_ap'):
                self.baseline_ap = current_ap
                self.text_logger.write(f"📊 Baseline detector mAP (epoch {epoch}): {self.baseline_ap:.4f}")

            # Show improvement from baseline
            improvement = current_ap - self.baseline_ap
            improvement_pct = (improvement / self.baseline_ap * 100) if self.baseline_ap > 0 else 0
            self.text_logger.write(
                f"📈 Current mAP: {current_ap:.4f} | Baseline: {self.baseline_ap:.4f} | "
                f"Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)"
            )

            if current_ap > self.best_ap:
                self.best_ap = current_ap
                self.text_logger.write(f"🏆 New best mAP: {self.best_ap:.4f} (epoch {epoch})")

                # Save best checkpoint
                best_checkpoint = {
                    "epoch": epoch,
                    "best_ap": self.best_ap,
                    "baseline_ap": self.baseline_ap,
                    "opt": self.opt,
                    "net_sr": self.get_bare_model(self.net_sr).state_dict(),
                    "net_det": self.get_bare_model(self.net_det).state_dict(),
                }
                save_on_master(self.get_bare_model(self.net_sr).state_dict(),
                              osp.join(self.exp_dir, 'models', "net_sr_best.pth"))
                save_on_master(self.get_bare_model(self.net_det).state_dict(),
                              osp.join(self.exp_dir, 'models', "net_det_best.pth"))
                save_on_master(best_checkpoint,
                              osp.join(self.exp_dir, 'checkpoints', "checkpoint_best.pth"))

                # Log best metric to wandb
                self.wandb_log({"val/det/best_mAP": self.best_ap}, step=wandb_step)

        return

    def save(self, epoch):            
        checkpoint = {"epoch": epoch,
                      "opt": self.opt,
                      "net_sr": self.get_bare_model(self.net_sr).state_dict(),
                      "net_det": self.get_bare_model(self.net_det).state_dict(),
                      'schedulers': [],
                      }
        for s in self.schedulers:
            checkpoint['schedulers'].append(s.state_dict())
                
        if epoch % self.opt['train']['save_freq'] == 0:
            save_on_master(self.get_bare_model(self.net_sr).state_dict(), osp.join(self.exp_dir, 'models', "net_sr_{:03d}.pth".format(epoch)))
            save_on_master(self.get_bare_model(self.net_det).state_dict(), osp.join(self.exp_dir, 'models', "net_det_{:03d}.pth".format(epoch)))
            save_on_master(checkpoint, osp.join(self.exp_dir, 'checkpoints', "checkpoint_{:03d}.pth".format(epoch)))
            
        save_on_master(self.get_bare_model(self.net_sr).state_dict(), osp.join(self.exp_dir, 'models', "net_sr_latest.pth"))
        save_on_master(self.get_bare_model(self.net_det).state_dict(), osp.join(self.exp_dir, 'models', "net_det_latest.pth"))
        save_on_master(checkpoint, osp.join(self.exp_dir, 'checkpoints', "checkpoint_latest.pth"))
        return
