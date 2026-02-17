import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import digamma


def window_partition(x, window_size: int):
    """
    x: [B, C, H, W]
    return: [B * num_windows, C, window_size, window_size]
    """
    B, C, H, W = x.shape
    assert H % window_size == 0 and W % window_size == 0, \
        f"H,W({H},{W}) must be divisible by window_size={window_size}"

    x = x.view(
        B,
        C,
        H // window_size, window_size,
        W // window_size, window_size
    )  # [B, C, H_ws, ws, W_ws, ws]

    # [B, H_ws, W_ws, C, ws, ws] -> [B*H_ws*W_ws, C, ws, ws]
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    windows = x.view(-1, C, window_size, window_size)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    windows: [B * num_windows, C, window_size, window_size]
    return: [B, C, H, W]
    """
    B_times_nw, C, ws, _ = windows.shape
    H_ws = H // window_size
    W_ws = W // window_size
    B = B_times_nw // (H_ws * W_ws)

    x = windows.view(B, H_ws, W_ws, C, ws, ws)  # [B, H_ws, W_ws, C, ws, ws]
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # [B, C, H_ws, ws, W_ws, ws]
    x = x.view(B, C, H, W)
    return x


class MHSA2D(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        HW, dim_head = H*W, C // self.heads
        qkv = self.qkv(x).reshape(B, 3, self.heads, dim_head, HW).transpose(-2, -1)

        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each: [B, heads, HW, dim_head]
        # attn = F.scaled_dot_product_attention(q, k, v)

        # TensorRT 변환 문제로 3D로 평탄화
        q = (q * self.scale).contiguous().view(B*self.heads, HW, dim_head)
        k = k.contiguous().view(B*self.heads, HW, dim_head)
        v = v.contiguous().view(B*self.heads, HW, dim_head)

        # scores: [B*heads, HW, HW]
        scores = torch.bmm(q, k.transpose(1, 2))
        scores = scores - scores.amax(dim=-1, keepdim=True)

        # TensorRT 변환 문제로 softmax 함수 대신 직접 연산
        exp_scores = torch.exp(scores)
        weights = exp_scores / exp_scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        # context: [B*heads, HW, dim_head] → [B, heads, HW, dim_head] → [B, C, H, W]
        attn = torch.bmm(weights, v).view(B, self.heads, HW, dim_head).transpose(2, 3).reshape(B, C, H, W)

        return self.proj(attn)
    

class WindowMHSA2D(nn.Module):
    def __init__(self, dim, heads=4, window_size=8, shift_size=0):
        """
        dim: 채널 수 C
        heads: multi-head 개수
        window_size: 윈도우 한 변 크기 (예: 8 -> 8x8 window)
        shift_size: >0이면 간단한 cyclic shift 적용 (옵션, 기본 0)
        """
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        ws = self.window_size

        # --- (옵션) 간단한 cyclic shift ---
        # Swin의 SW-MSA처럼 완전 동일한 구현은 아니고,
        # 나중에 필요하면 여기서 mask 추가해서 정교하게 확장하면 됨.
        if self.shift_size > 0:
            shift = self.shift_size
            x = torch.roll(x, shifts=(-shift, -shift), dims=(2, 3))

        # 1) window로 쪼개기
        x_windows = window_partition(x, ws)  # [B*num_windows, C, ws, ws]
        Bn, C, ws, _ = x_windows.shape
        N = ws * ws  # window 당 토큰 수

        # 2) 각 window에 대해 qkv → attention
        qkv = self.qkv(x_windows)  # [Bn, 3C, ws, ws]
        qkv = qkv.reshape(Bn, 3, self.heads, C // self.heads, N)
        # [Bn, 3, heads, dim_head, N] -> [3, Bn, heads, N, dim_head]
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 각: [Bn, heads, N, dim_head]

        q = (q * self.scale).contiguous().view(Bn * self.heads, N, -1)
        k = k.contiguous().view(Bn * self.heads, N, -1)
        v = v.contiguous().view(Bn * self.heads, N, -1)

        scores = torch.bmm(q, k.transpose(1, 2))  # [Bn*heads, N, N]
        scores = scores - scores.amax(dim=-1, keepdim=True)
        exp_scores = torch.exp(scores)
        weights = exp_scores / exp_scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        out = torch.bmm(weights, v)  # [Bn*heads, N, dim_head]
        out = out.view(Bn, self.heads, N, C // self.heads)
        out = out.permute(0, 1, 3, 2).contiguous().view(Bn, C, ws, ws)

        out = self.proj(out)  # [Bn, C, ws, ws]

        # 3) window들을 다시 합치기
        out = window_reverse(out, ws, H, W)  # [B, C, H, W]

        # 4) (옵션) shift 되돌리기
        if self.shift_size > 0:
            shift = self.shift_size
            out = torch.roll(out, shifts=(shift, shift), dims=(2, 3))

        return out
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, attn_type="none"):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.attn_type = attn_type

        if attn_type == "global":
            self.attn = MHSA2D(in_channels, heads=4)
        elif attn_type == "window":
            self.attn = WindowMHSA2D(in_channels, heads=4, window_size=8, shift_size=0)
        else:
            self.attn = None


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Skip Connection
        out = F.relu(out)

        if self.attn is not None:
            out = out + self.attn(out)

        return out

class UpsampleBlock(nn.Module):
    def __init__(self, base_channels, num_residual_blocks=2, scale_factor=2, upsample_type='bilinear'):
        super(UpsampleBlock, self).__init__()
        self.scale_factor = scale_factor
        self.res_blocks = nn.Sequential(*[ResidualBlock(base_channels, attn_type="none") for _ in range(num_residual_blocks)])
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.res_blocks(x)
        return x

class UpsampleResNet(nn.Module):
    def __init__(self, in_channels=1, num_residual_blocks=8, base_channels=64):
        super(UpsampleResNet, self).__init__()

        self.initial_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

        self.res_blocks = nn.Sequential(*[ResidualBlock(base_channels, attn_type="none") for _ in range(num_residual_blocks)])
        self.low_proj_conv = nn.Conv2d(base_channels + 1, base_channels, kernel_size=3, stride=1, padding=1)
        self.upsample1 = UpsampleBlock(base_channels, scale_factor=2)
        self.upsample2 = UpsampleBlock(base_channels, scale_factor=2)
        self.upsample3 = UpsampleBlock(base_channels, scale_factor=2)

        self.low_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        down_x = self.low_conv(x)
        x = torch.cat([down_x, x], dim=1)
        x = self.low_proj_conv(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.final_conv(x)

        return x, down_x
    
# Denoising ResNet, with initial conv, several residual blocks with attention, and final conv
class DenoiseResNet(nn.Module):
    def __init__(self, in_channels=1, num_residual_blocks=8, base_channels=64, global_residual=False, attn_type="window"):
        super(DenoiseResNet, self).__init__()

        self.initial_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(base_channels, attn_type=attn_type) for _ in range(num_residual_blocks)])
        self.low_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.global_residual = global_residual

    def forward(self, x):
        identity = x
        out = self.initial_conv(x)
        out = self.res_blocks(out)
        out = self.low_conv(out)
        if self.global_residual:
            out = identity + out
        return out


class Upsampler(nn.Module):
    def __init__(self, n_feats, scale=8, upsample_type='pixelshuffle'):
        super().__init__()
        assert scale in [2, 4, 8], "지원하는 scale은 2,4,8 정도로 제한하는 걸 추천"
        m = []
        if upsample_type == 'pixelshuffle':
            for _ in range(3):
                # 채널을 4배로 늘리고 shuffle로 2배 업샘플
                m += [
                    nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
                    nn.PixelShuffle(2),
                    nn.ReLU(inplace=True),
                ]
        elif upsample_type == 'bilinear':
            for _ in range(3):
                m += [
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                ]
        else:
            raise ValueError(f"Unsupported upsample_type: {upsample_type}")

        self.upsampler = nn.Sequential(*m)

    def forward(self, x):
        return self.upsampler(x)


# Super-Resolution ResNet, with low projection conv, several upsample blocks, and final conv
# TODO: add pixel shuffle for upsampling
# class SRResNet(nn.Module):
#     def __init__(self, in_channels=1, base_channels=64, upsample_type='pixelshuffle', use_bicubic_residual=True):
#         super(SRResNet, self).__init__()
#         self.low_proj_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
#         self.upsample1 = UpsampleBlock(base_channels, scale_factor=2, upsample_type=upsample_type)
#         self.upsample2 = UpsampleBlock(base_channels, scale_factor=2, upsample_type=upsample_type)
#         self.upsample3 = UpsampleBlock(base_channels, scale_factor=2, upsample_type=upsample_type)
#         self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         x = self.low_proj_conv(x)
#         x = self.upsample1(x)
#         x = self.upsample2(x)
#         x = self.upsample3(x)
#         x = self.final_conv(x)
#         return x


class SRResNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        base_channels=64,
        num_resblocks=16,
        upsample_type='pixelshuffle',
        use_bicubic_residual=True,
        scale=8,
        attn_type="none",
    ):
        super().__init__()
        self.scale = scale
        self.use_bicubic_residual = use_bicubic_residual
        print("[SRResNet] upsample_type:", upsample_type)
        print("[SRResNet] use_bicubic_residual:", use_bicubic_residual)

        self.head = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

        body = []
        for _ in range(num_resblocks):
            body.append(ResidualBlock(base_channels, attn_type=attn_type))
        body.append(nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1))
        self.body = nn.Sequential(*body)

        self.upsampler = Upsampler(base_channels, scale=scale, upsample_type=upsample_type)
        self.tail = nn.Conv2d(base_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        x: [B,1,H,W] (denoiser output 저해상도, 예: 32x32)
        output: [B,1,H*scale,W*scale]
        """
        # 선택: bicubic global skip (학습 안정 + coarse 구조 보존)
        if self.use_bicubic_residual:
            x_bicubic = F.interpolate(
                x, scale_factor=self.scale, mode='bilinear', align_corners=False
            )
        else:
            x_bicubic = None

        # Head
        feat = self.head(x)

        # Body (residual-in-residual 구조는 단순화해서 feat + body(feat))
        res = self.body(feat)
        feat = feat + res   # global residual in feature space

        # Upsample + Tail
        up = self.upsampler(feat)   # [B,C,H*8,W*8]
        out = self.tail(up)         # [B,1,H*8,W*8]

        if x_bicubic is not None:
            out = out + x_bicubic

        return out


# Combined Denoising and Super-Resolution Model
class SonarSRModel(nn.Module):
    def __init__(self, in_channels=1, num_residual_blocks=8, base_channels=64, 
                 denoiser_global_residual=False, sr_upsample_type='bilinear', sr_use_bicubic_residual=False,
                 denoiser_attn_type="window", sr_attn_type="none"):
        super(SonarSRModel, self).__init__()
        self.denoiser = DenoiseResNet(in_channels=in_channels, 
                                      num_residual_blocks=num_residual_blocks, 
                                      base_channels=base_channels,
                                      global_residual=denoiser_global_residual,
                                      attn_type=denoiser_attn_type)
        self.upsampler = SRResNet(in_channels=in_channels, base_channels=base_channels, upsample_type=sr_upsample_type, use_bicubic_residual=sr_use_bicubic_residual, attn_type=sr_attn_type)

    def forward(self, x):
        denoised_x = self.denoiser(x)
        final_x = self.upsampler(denoised_x)
        return final_x, denoised_x

def soft_gamma_nll_loss(y, x_hat, enl, eps=1e-3):
    x_hat = 0.5 * (x_hat.clamp(min=-1) + 1)
    y = 0.5 * (y.clamp(min=-1) + 1)
    # Gamma NLL loss (up to constant terms)
    loss_map = torch.log(x_hat + eps) + y / (x_hat + eps)
    return loss_map.mean() * enl

def log_mse_loss(gt, pred, eps=1e-3):
    pred = 0.5 * (pred.clamp(min=-1) + 1)
    gt = 0.5 * (gt.clamp(min=-1) + 1)
    # Log-domain MSE
    loss = F.mse_loss(torch.log(pred + eps), torch.log(gt + eps), reduction='mean')
    return loss

def convert_to_buffer(module: torch.nn.Module, persistent: bool = True):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)


def build_network(
    in_channels=1,
    num_residual_blocks=8,
    base_channels=64,
    scale=8,
    # 체크포인트에 따라 설정
    denoiser_global_residual=True,
    sr_upsample_type='pixelshuffle',
    sr_use_bicubic_residual=True,
    denoiser_attn_type="global",  # sr_arch_pixelshuffle_globalo_v2 uses global
    sr_attn_type="none",
    **kwargs
):
    """
    Build SonarSR network for SR4IR

    Checkpoint variants:
    - sr_arch_pixelshuffle_globalo_v2: denoiser_attn_type="global", sr_upsample_type="pixelshuffle"
    - windowattn_BLresidual: denoiser_attn_type="window", sr_use_bicubic_residual=True
    """
    if scale != 8:
        raise ValueError(f"SonarSR only supports scale=8, got scale={scale}")

    return SonarSRModel(
        in_channels=in_channels,
        num_residual_blocks=num_residual_blocks,
        base_channels=base_channels,
        denoiser_global_residual=denoiser_global_residual,
        sr_upsample_type=sr_upsample_type,
        sr_use_bicubic_residual=sr_use_bicubic_residual,
        denoiser_attn_type=denoiser_attn_type,
        sr_attn_type=sr_attn_type,
    )