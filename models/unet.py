# -*- coding: utf-8 -*-
"""
U-Net 语义分割模型
===================
实现标准 U-Net 结构：编码器(4次下采样) -> 瓶颈层 -> 解码器(4次上采样) + 跳跃连接。
支持二分类(single channel output)和多分类(multi-channel output)。

参考论文: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", 2015.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """双卷积块：(Conv2d -> BatchNorm -> ReLU) * 2
    U-Net中每个层级的基本构建单元。
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """下采样模块：MaxPool2d -> 双卷积块
    用于编码器阶段，逐步降低空间分辨率、增加通道数。
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """上采样模块：上采样 -> 拼接跳跃连接 -> 双卷积块
    用于解码器阶段，逐步恢复空间分辨率。
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            # 使用双线性插值上采样，减少通道数
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            # 使用转置卷积上采样
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Args:
            x1: 解码器上一层输出(待上采样)
            x2: 对应编码器层输出(跳跃连接)
        """
        x1 = self.up(x1)
        # 处理尺寸不匹配的情况(对齐填充)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # 在通道维度拼接跳跃连接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """输出层：1x1卷积，输出通道数为类别数"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net 语义分割网络

    结构:
        编码器: 4次下采样 (64 -> 128 -> 256 -> 512 -> 1024)
        瓶颈层: 1024通道
        解码器: 4次上采样 (1024 -> 512 -> 256 -> 128 -> 64)
        输出层: 1x1卷积映射到类别数

    Args:
        in_channels: 输入通道数(RGB图像为3)
        num_classes: 类别数。二分类时为1(单通道输出)，多分类时为类别数
        bilinear: 是否使用双线性插值上采样(否则用转置卷积)
        base_features: 初始特征数(默认64)
    """

    def __init__(self, in_channels=3, num_classes=1, bilinear=True, base_features=64):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        f = base_features  # 初始特征数
        # 双线性上采样不改变通道数，需用factor缩减瓶颈层与解码器输出，
        # 使拼接后通道数恰好等于Up模块的in_channels
        factor = 2 if bilinear else 1

        # 编码器(下采样路径)
        self.inc = DoubleConv(in_channels, f)            # 64
        self.down1 = Down(f, f * 2)                      # 128
        self.down2 = Down(f * 2, f * 4)                  # 256
        self.down3 = Down(f * 4, f * 8)                  # 512
        self.down4 = Down(f * 8, f * 16 // factor)       # 瓶颈层(bilinear时为512)

        # 解码器(上采样路径)
        # Up的in_channels = 拼接后总通道数 = x1通道 + x2通道
        self.up1 = Up(f * 16, f * 8 // factor, bilinear)  # 512
        self.up2 = Up(f * 8, f * 4 // factor, bilinear)   # 256
        self.up3 = Up(f * 4, f * 2 // factor, bilinear)    # 128
        self.up4 = Up(f * 2, f, bilinear)                  # 64

        # 输出层
        self.outc = OutConv(f, num_classes)

    def forward(self, x):
        # 编码器：保存各层特征用于跳跃连接
        x1 = self.inc(x)      # [B, 64, H, W]
        x2 = self.down1(x1)   # [B, 128, H/2, W/2]
        x3 = self.down2(x2)   # [B, 256, H/4, W/4]
        x4 = self.down3(x3)   # [B, 512, H/8, W/8]
        x5 = self.down4(x4)   # [B, 1024, H/16, W/16] 瓶颈层

        # 解码器：上采样并拼接对应的编码器特征
        x = self.up1(x5, x4)  # [B, 512, H/8, W/8]
        x = self.up2(x, x3)   # [B, 256, H/4, W/4]
        x = self.up3(x, x2)   # [B, 128, H/2, W/2]
        x = self.up4(x, x1)   # [B, 64, H, W]
        logits = self.outc(x) # [B, num_classes, H, W]
        return logits

    def __repr__(self):
        return (f"UNet(in_channels={self.in_channels}, "
                f"num_classes={self.num_classes}, "
                f"bilinear={self.bilinear})")
