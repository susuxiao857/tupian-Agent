# -*- coding: utf-8 -*-
"""
数据增强策略集
==================
实现水平/垂直翻转、随机旋转、弹性变形等数据增强。
支持配置开关，可被训练流程调用。
"""

import random

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class SegmentationAugmentor:
    """分割任务数据增强器

    对图像和标签同时进行相同的几何变换，保证空间对齐。
    所有增强均可通过配置开关独立控制。

    Args:
        aug_config: 增强配置字典，包含各增强策略的开关
        prob: 每种增强被应用的概率(0-1)
    """

    def __init__(self, aug_config=None, prob=0.5):
        default_config = {
            "horizontal_flip": False,
            "vertical_flip": False,
            "random_rotation": False,
            "elastic_deform": False,
        }
        if aug_config:
            default_config.update(aug_config)
        self.aug_config = default_config
        self.prob = prob

    def __call__(self, image, mask):
        """应用数据增强

        Args:
            image: numpy数组 [H, W, C]，图像
            mask: numpy数组 [H, W]，标签

        Returns:
            image: 增强后的图像
            mask: 增强后的标签
        """
        # 水平翻转
        if self.aug_config.get("horizontal_flip", False) and random.random() < self.prob:
            image, mask = self._horizontal_flip(image, mask)

        # 垂直翻转
        if self.aug_config.get("vertical_flip", False) and random.random() < self.prob:
            image, mask = self._vertical_flip(image, mask)

        # 随机旋转
        if self.aug_config.get("random_rotation", False) and random.random() < self.prob:
            image, mask = self._random_rotation(image, mask)

        # 弹性变形
        if self.aug_config.get("elastic_deform", False) and random.random() < self.prob:
            image, mask = self._elastic_deform(image, mask)

        return image, mask

    # ==================== 增强方法 ====================
    def _horizontal_flip(self, image, mask):
        """水平翻转"""
        return np.fliplr(image).copy(), np.fliplr(mask).copy()

    def _vertical_flip(self, image, mask):
        """垂直翻转"""
        return np.flipud(image).copy(), np.flipud(mask).copy()

    def _random_rotation(self, image, mask, angle_range=(-30, 30)):
        """随机旋转

        Args:
            image: 输入图像
            mask: 输入标签
            angle_range: 旋转角度范围(度)
        """
        angle = random.uniform(*angle_range)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        if HAS_CV2:
            # 使用OpenCV旋转变换
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            # 图像用双线性插值，标签用最近邻插值
            image_rot = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
            mask_rot = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
            return image_rot, mask_rot
        else:
            # 回退方案：使用numpy简单实现90度整数倍旋转
            k = int(angle / 90) % 4
            return (
                np.rot90(image, k).copy(),
                np.rot90(mask, k).copy(),
            )

    def _elastic_deform(self, image, mask, alpha=120, sigma=12):
        """弹性变形

        通过对位移场施加高斯平滑产生局部弹性形变。
        参考: Simard et al., "Best Practices for Convolutional Neural Networks
              Applied to Visual Document Analysis", 2003.

        Args:
            image: 输入图像
            mask: 输入标签
            alpha: 变形强度
            sigma: 高斯平滑标准差
        """
        if not HAS_CV2:
            # 无OpenCV时跳过弹性变形
            return image, mask

        h, w = image.shape[:2]

        # 生成随机位移场
        dx = np.random.uniform(-1, 1, (h, w)).astype(np.float32)
        dy = np.random.uniform(-1, 1, (h, w)).astype(np.float32)

        # 高斯平滑位移场
        ksize = max(3, int(sigma * 6) | 1)  # 确保奇数
        dx = cv2.GaussianBlur(dx, (ksize, ksize), sigma) * alpha
        dy = cv2.GaussianBlur(dy, (ksize, ksize), sigma) * alpha

        # 构建映射网格
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        # 应用重映射
        image_deformed = cv2.remap(image, map_x, map_y,
                                  interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)
        mask_deformed = cv2.remap(mask, map_x, map_y,
                                  interpolation=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_REFLECT)

        return image_deformed, mask_deformed

    def update_config(self, aug_config):
        """更新增强配置

        Args:
            aug_config: 新的增强配置字典
        """
        self.aug_config.update(aug_config)

    def get_config(self):
        """获取当前增强配置"""
        return self.aug_config.copy()
