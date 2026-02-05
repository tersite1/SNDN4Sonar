# ------------------------------------------------------------------------
# SonarSR Model - EXACT COPY from original model.py
# Pretrained weight compatible
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, use_attention=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.use_attn = use_attention

        if use_attention:
            self.mhsa = MHSA2D(in_channels, heads=4)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)

        if self.use_attn:
            out = out + self.mhsa(out)

        return out


class UpsampleBlock(nn.Module):
    def __init__(self, base_channels, num_residual_blocks=2, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.scale_factor = scale_factor
        self.res_blocks = nn.Sequential(*[ResidualBlock(base_channels, use_attention=False) for _ in range(num_residual_blocks)])

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.res_blocks(x)
        return x


class UpsampleResNet(nn.Module):
    """
    Original SonarSR Model - EXACT architecture

    Input: [B, 1, 80, 80] noisy LR
    Output: ([B, 1, 640, 640] SR, [B, 1, 80, 80] denoised LR)
    """
    def __init__(self, in_channels=1, num_residual_blocks=8, base_channels=64):
        super(UpsampleResNet, self).__init__()

        self.initial_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

        # Denoising stage with attention
        self.res_blocks = nn.Sequential(*[ResidualBlock(base_channels, use_attention=True) for _ in range(num_residual_blocks)])

        # Denoised LR output
        self.low_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, stride=1, padding=1)

        # Concat projection
        self.low_proj_conv = nn.Conv2d(base_channels + in_channels, base_channels, kernel_size=3, stride=1, padding=1)

        # 3x bilinear upsample (2x2x2 = 8x)
        self.upsample1 = UpsampleBlock(base_channels, scale_factor=2)
        self.upsample2 = UpsampleBlock(base_channels, scale_factor=2)
        self.upsample3 = UpsampleBlock(base_channels, scale_factor=2)

        # Final SR output
        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Args:
            x: [B, 1, 80, 80] noisy LR

        Returns:
            final_sr: [B, 1, 640, 640] clean SR
            denoised_lr: [B, 1, 80, 80] clean LR
        """
        x = self.initial_conv(x)
        x = self.res_blocks(x)

        # Denoised LR output
        down_x = self.low_conv(x)

        # Concat and project
        x = torch.cat([down_x, x], dim=1)
        x = self.low_proj_conv(x)

        # Upsample 8x (2x2x2)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)

        # Final SR output
        x = self.final_conv(x)

        return x, down_x


def build_network(
    in_channels=1,
    num_residual_blocks=8,
    base_channels=64,
    scale=8,
    **kwargs
):
    """
    Build original SonarSR network for SR4IR

    NOTE: All other parameters (denoiser_global_residual, sr_upsample_type, etc.)
          are IGNORED because this is the exact original architecture.

    Config example:
        network_sr:
          name: sonarsr
          in_channels: 1
          base_channels: 64
          num_residual_blocks: 8
          scale: 8
    """
    if scale != 8:
        raise ValueError(f"Original SonarSR only supports scale=8, got scale={scale}")

    return UpsampleResNet(
        in_channels=in_channels,
        num_residual_blocks=num_residual_blocks,
        base_channels=base_channels,
    )
