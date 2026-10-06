# -*- coding: utf-8 -*-
"""
分割评估指标计算
==================
实现 IoU(Jaccard)、Dice(F1)、Pixel Accuracy 等分割评估指标。
支持 batch 级别计算，兼容二分类与多分类。

指标说明:
    - IoU (Intersection over Union / Jaccard): 交并比，衡量预测与真实的重叠度
    - Dice (F1 Score): 与IoU正相关，对边界更敏感
    - Pixel Accuracy: 像素级准确率，最直观但类别不平衡时不准确
"""

import numpy as np
import torch


def _check_tensor(pred, target):
    """校验输入张量并转为numpy数组"""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    return pred, target


def _get_pred_labels(pred, num_classes):
    """将模型输出(logits或概率)转为类别标签"""
    if isinstance(pred, torch.Tensor):
        # 如果是多通道输出，取argmax
        if pred.ndim == 4 and pred.shape[1] > 1:
            pred = torch.argmax(pred, dim=1)
        elif pred.ndim == 4 and pred.shape[1] == 1:
            pred = (torch.sigmoid(pred) > 0.5).long()
        pred = pred.detach().cpu().numpy()
    else:
        pred = np.asarray(pred)
    return pred


def iou_score(pred, target, num_classes=2, smooth=1e-6, ignore_index=None):
    """计算 IoU (Jaccard Index)

    IoU = |P ∩ G| / |P ∪ G| = TP / (TP + FP + FN)

    Args:
        pred: 预测结果，可为logits[B,C,H,W]或标签[B,H,W]
        target: 真实标签[B,H,W]
        num_classes: 类别数(二分类时为2: 背景+前景)
        smooth: 平滑项，避免除零
        ignore_index: 忽略的标签值(如255表示无效像素)

    Returns:
        float: 平均IoU(各类别取平均)
        list: 各类别IoU
    """
    pred_labels = _get_pred_labels(pred, num_classes)
    target = target.detach().cpu().numpy() if isinstance(target, torch.Tensor) else np.asarray(target)

    # 展平
    pred_flat = pred_labels.flatten()
    target_flat = target.flatten()

    # 过滤无效像素
    if ignore_index is not None:
        valid = target_flat != ignore_index
        pred_flat = pred_flat[valid]
        target_flat = target_flat[valid]

    iou_per_class = []
    for cls in range(num_classes):
        pred_cls = pred_flat == cls
        target_cls = target_flat == cls
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        iou = (intersection + smooth) / (union + smooth)
        iou_per_class.append(float(iou))

    mean_iou = float(np.mean(iou_per_class))
    return mean_iou, iou_per_class


def dice_score(pred, target, num_classes=2, smooth=1e-6, ignore_index=None):
    """计算 Dice 系数 (F1 Score)

    Dice = 2|P ∩ G| / (|P| + |G|) = 2*TP / (2*TP + FP + FN)

    Args:
        pred: 预测结果
        target: 真实标签
        num_classes: 类别数
        smooth: 平滑项
        ignore_index: 忽略的标签值

    Returns:
        float: 平均Dice系数
        list: 各类别Dice系数
    """
    pred_labels = _get_pred_labels(pred, num_classes)
    target = target.detach().cpu().numpy() if isinstance(target, torch.Tensor) else np.asarray(target)

    pred_flat = pred_labels.flatten()
    target_flat = target.flatten()

    if ignore_index is not None:
        valid = target_flat != ignore_index
        pred_flat = pred_flat[valid]
        target_flat = target_flat[valid]

    dice_per_class = []
    for cls in range(num_classes):
        pred_cls = pred_flat == cls
        target_cls = target_flat == cls
        intersection = np.logical_and(pred_cls, target_cls).sum()
        denom = pred_cls.sum() + target_cls.sum()
        dice = (2.0 * intersection + smooth) / (denom + smooth)
        dice_per_class.append(float(dice))

    mean_dice = float(np.mean(dice_per_class))
    return mean_dice, dice_per_class


def pixel_accuracy(pred, target, num_classes=2, ignore_index=None):
    """计算像素准确率

    PA = TP / (TP + FP + FN) = 正确像素数 / 总像素数

    Args:
        pred: 预测结果
        target: 真实标签
        num_classes: 类别数
        ignore_index: 忽略的标签值

    Returns:
        float: 像素准确率
    """
    pred_labels = _get_pred_labels(pred, num_classes)
    target = target.detach().cpu().numpy() if isinstance(target, torch.Tensor) else np.asarray(target)

    pred_flat = pred_labels.flatten()
    target_flat = target.flatten()

    if ignore_index is not None:
        valid = target_flat != ignore_index
        pred_flat = pred_flat[valid]
        target_flat = target_flat[valid]

    correct = (pred_flat == target_flat).sum()
    total = pred_flat.size
    if total == 0:
        return 0.0
    return float(correct) / float(total)


def compute_all_metrics(pred, target, num_classes=2, ignore_index=None, smooth=1e-6):
    """一次性计算所有指标

    Args:
        pred: 预测结果
        target: 真实标签
        num_classes: 类别数
        ignore_index: 忽略的标签值
        smooth: 平滑项

    Returns:
        dict: 包含 iou, dice, pixel_accuracy 及各类别细分
    """
    mean_iou, iou_per_class = iou_score(
        pred, target, num_classes, smooth, ignore_index
    )
    mean_dice, dice_per_class = dice_score(
        pred, target, num_classes, smooth, ignore_index
    )
    pa = pixel_accuracy(pred, target, num_classes, ignore_index)

    return {
        "iou": mean_iou,
        "iou_per_class": iou_per_class,
        "dice": mean_dice,
        "dice_per_class": dice_per_class,
        "pixel_accuracy": pa,
    }
