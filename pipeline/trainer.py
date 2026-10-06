# -*- coding: utf-8 -*-
"""
U-Net 训练流程封装
===================
将完整的 U-Net 训练流程封装为 UNetTrainer 类，作为可调度单元被 Agent 调用。
内部实现前向传播、损失计算、反向传播、优化器更新等完整训练循环。
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.unet import UNet
from utils.metrics import compute_all_metrics
from utils.logger import ExperimentLogger


class SegmentationDataset(Dataset):
    """分割数据集(内存版)

    将图像和标签加载到内存，支持数据增强。

    Args:
        data: 数据列表，每项为 (image_array, label_array)
        augmentor: 数据增强器(可选)
        is_train: 是否为训练集(决定是否应用增强)
    """

    def __init__(self, data, augmentor=None, is_train=True):
        self.data = data
        self.augmentor = augmentor
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        # 转为numpy(H, W, C)和(H, W)
        image = np.array(image, dtype=np.float32)
        label = np.array(label, dtype=np.int64)

        # 保证图像至少3维
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        # 应用数据增强(仅训练阶段)
        if self.is_train and self.augmentor is not None:
            image, label = self.augmentor(image, label)

        # 转为tensor: [C, H, W] 和 [H, W]
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        label = torch.from_numpy(label).long()

        return image, label


class UNetTrainer:
    """U-Net 训练器

    封装完整训练循环，作为可调度单元被 Agent 调用。

    Args:
        config: 超参配置字典
        num_classes: 类别数
        in_channels: 输入通道数
        device: 计算设备
        logger: 日志记录器
        init_state: 可选，上一轮最佳权重(热启动)
        base_features: U-Net 初始通道数
        grad_clip_norm: 梯度裁剪阈值，None/0 表示不裁剪
    """

    def __init__(self, config, num_classes=2, in_channels=3,
                 device="cpu", logger=None, init_state=None,
                 base_features=64, grad_clip_norm=1.0):
        self.config = config
        self.num_classes = num_classes
        self.in_channels = in_channels
        use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.logger = logger or ExperimentLogger()
        self.grad_clip_norm = grad_clip_norm
        self.model_best_state = None

        self.model = UNet(
            in_channels=in_channels,
            num_classes=1 if num_classes == 2 else num_classes,
            bilinear=True,
            base_features=base_features,
        ).to(self.device)

        if init_state is not None:
            self.model.load_state_dict(init_state)
            self.logger.info("[训练] 已加载上一轮最佳权重(热启动)")

        self.criterion = self._build_criterion(config.get("loss_function", "dice"))
        self.epoch_metrics = []

    def _build_criterion(self, loss_name):
        """构建损失函数"""
        if loss_name == "dice":
            return self._dice_loss
        elif loss_name == "ce":
            return nn.CrossEntropyLoss()
        elif loss_name == "dice_ce":
            ce = nn.CrossEntropyLoss()
            return lambda pred, target: ce(pred, target) + self._dice_loss(pred, target)
        else:
            return self._dice_loss

    def _dice_loss(self, pred, target, smooth=1e-6):
        """Dice损失

        Args:
            pred: 模型输出 logits [B, C, H, W]
            target: 真实标签 [B, H, W]
            smooth: 平滑项
        """
        if pred.shape[1] == 1:
            # 二分类：用sigmoid
            pred_prob = torch.sigmoid(pred)
            target_f = target.float().unsqueeze(1)
        else:
            # 多分类：用softmax
            pred_prob = torch.softmax(pred, dim=1)
            target_f = torch.nn.functional.one_hot(target, self.num_classes)
            target_f = target_f.permute(0, 3, 1, 2).float()

        intersection = (pred_prob * target_f).sum(dim=(0, 2, 3))
        cardinality = pred_prob.sum(dim=(0, 2, 3)) + target_f.sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + smooth) / (cardinality + smooth)
        return 1.0 - dice.mean()

    def _build_optimizer(self, lr):
        """构建优化器"""
        opt_name = self.config.get("optimizer", "adam").lower()
        weight_decay = self.config.get("weight_decay", 1e-5)

        if opt_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr,
                                    weight_decay=weight_decay)
        elif opt_name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=lr,
                                   momentum=0.9, weight_decay=weight_decay)
        else:
            return torch.optim.Adam(self.model.parameters(), lr=lr,
                                    weight_decay=weight_decay)

    def train(self, train_data, val_data=None, augmentor=None):
        """执行训练流程

        Args:
            train_data: 训练数据列表 [(image, label), ...]
            val_data: 验证数据列表(可选)
            augmentor: 数据增强器

        Returns:
            dict: 训练结果，包含模型状态、指标
        """
        lr = self.config.get("learning_rate", 1e-3)
        batch_size = self.config.get("batch_size", 4)
        num_epochs = self.config.get("num_epochs", 5)

        self.logger.info(
            f"开始训练: lr={lr}, batch_size={batch_size}, "
            f"epochs={num_epochs}, device={self.device}"
        )

        optimizer = self._build_optimizer(lr)

        pin_memory = self.device.type == "cuda"
        effective_bs = max(1, min(int(batch_size), max(len(train_data), 1)))
        train_dataset = SegmentationDataset(train_data, augmentor, is_train=True)
        train_loader = DataLoader(
            train_dataset, batch_size=effective_bs, shuffle=True,
            drop_last=False, num_workers=0, pin_memory=pin_memory,
        )

        val_loader = None
        if val_data:
            val_dataset = SegmentationDataset(val_data, augmentor=None, is_train=False)
            val_bs = max(1, min(int(batch_size), max(len(val_data), 1)))
            val_loader = DataLoader(
                val_dataset, batch_size=val_bs, shuffle=False,
                drop_last=False, num_workers=0, pin_memory=pin_memory,
            )

        best_iou = -1.0
        best_metrics = {}
        avg_loss = 0.0
        val_metrics = {}

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for images, labels in train_loader:
                images = images.to(self.device, non_blocking=pin_memory)
                labels = labels.to(self.device, non_blocking=pin_memory)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                if self.grad_clip_norm:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"train_loss={avg_loss:.4f} | "
                    f"val_iou={val_metrics.get('iou', 0):.4f} | "
                    f"val_dice={val_metrics.get('dice', 0):.4f}"
                )
                if val_metrics.get("iou", 0) > best_iou:
                    best_iou = val_metrics["iou"]
                    best_metrics = val_metrics
                    self.model_best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
            else:
                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | train_loss={avg_loss:.4f}"
                )

        if self.model_best_state is not None:
            self.model.load_state_dict(self.model_best_state)
            self.logger.info(f"[训练] 已恢复最佳权重 (val IoU={best_iou:.4f})")
        elif val_loader is None:
            best_metrics = self.evaluate(train_loader)
            self.model_best_state = {
                k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()
            }

        return {
            "model": self.model,
            "model_state": self.model_best_state or {
                k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()
            },
            "best_metrics": best_metrics if best_metrics else val_metrics,
            "final_loss": avg_loss,
        }

    @torch.no_grad()
    def evaluate(self, loader):
        """在数据加载器上评估模型

        Args:
            loader: 数据加载器

        Returns:
            dict: 评估指标
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        for images, labels in loader:
            images = images.to(self.device, non_blocking=self.device.type == "cuda")
            labels = labels.to(self.device, non_blocking=self.device.type == "cuda")

            logits = self.model(images)
            preds = torch.argmax(logits, dim=1) if logits.shape[1] > 1 else \
                    (torch.sigmoid(logits) > 0.5).long().squeeze(1)

            all_preds.append(preds)
            all_targets.append(labels)

        # 合并所有batch，并移至CPU以便numpy计算
        all_preds = torch.cat(all_preds, dim=0).cpu()
        all_targets = torch.cat(all_targets, dim=0).cpu()

        metrics = compute_all_metrics(
            all_preds, all_targets,
            num_classes=self.num_classes,
        )
        return metrics
