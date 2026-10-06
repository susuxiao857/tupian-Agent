# -*- coding: utf-8 -*-
"""
指标评估模块
==================
在验证集上计算 IoU、Dice、像素准确率等分割评估指标，返回结构化结果。
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.metrics import compute_all_metrics
from utils.logger import ExperimentLogger


class Evaluator:
    """分割指标评估器

    在给定数据上评估模型，计算IoU/Dice/PA等指标。

    Args:
        model: 训练好的分割模型
        device: 计算设备
        num_classes: 类别数
        logger: 日志记录器
    """

    def __init__(self, model, device="cpu", num_classes=2, logger=None):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.num_classes = num_classes
        self.logger = logger or ExperimentLogger()

    @torch.no_grad()
    def evaluate(self, data, batch_size=4, return_predictions=False):
        """在数据上评估模型

        Args:
            data: 数据列表 [(image, label), ...] 或 DataLoader
            batch_size: 批大小(当data为列表时使用)
            return_predictions: 是否返回预测结果

        Returns:
            dict: 评估结果
                - metrics: 指标字典
                - predictions: 预测列表(return_predictions=True时)
        """
        self.model.to(self.device)
        self.model.eval()

        # 如果传入的是列表，构建DataLoader
        if isinstance(data, list):
            from pipeline.trainer import SegmentationDataset
            dataset = SegmentationDataset(data, augmentor=None, is_train=False)
            loader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, drop_last=False)
        else:
            loader = data

        all_preds = []
        all_targets = []
        predictions = []

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)

            # 转为预测标签
            if logits.shape[1] > 1:
                preds = torch.argmax(logits, dim=1)
            else:
                preds = (torch.sigmoid(logits) > 0.5).long().squeeze(1)

            all_preds.append(preds)
            all_targets.append(labels)

            if return_predictions:
                predictions.append(preds.cpu().numpy())

        # 合并所有batch，并移至CPU以便numpy计算
        all_preds = torch.cat(all_preds, dim=0).cpu()
        all_targets = torch.cat(all_targets, dim=0).cpu()

        # 计算指标
        metrics = compute_all_metrics(
            all_preds, all_targets,
            num_classes=self.num_classes,
        )

        # 结构化结果
        result = {
            "metrics": metrics,
            "num_samples": len(all_preds),
            "num_classes": self.num_classes,
            "device": str(self.device),
        }

        if return_predictions:
            result["predictions"] = predictions

        self.logger.info(
            f"评估完成 | 样本数={result['num_samples']} | "
            f"IoU={metrics['iou']:.4f} | "
            f"Dice={metrics['dice']:.4f} | "
            f"PA={metrics['pixel_accuracy']:.4f}"
        )

        return result

    def evaluate_and_compare(self, data, thresholds, batch_size=4):
        """评估并与阈值对比

        Args:
            data: 数据列表或DataLoader
            thresholds: 阈值字典，如 {"iou": 0.7, "dice": 0.75}
            batch_size: 批大小

        Returns:
            dict: 评估结果 + 是否达标
        """
        result = self.evaluate(data, batch_size=batch_size)
        metrics = result["metrics"]

        # 检查是否达标
        threshold_checks = {}
        all_passed = True
        for metric_name, threshold in thresholds.items():
            actual = metrics.get(metric_name, 0.0)
            passed = actual >= threshold
            threshold_checks[metric_name] = {
                "threshold": threshold,
                "actual": actual,
                "passed": passed,
                "margin": actual - threshold,
            }
            if not passed:
                all_passed = False

        result["threshold_checks"] = threshold_checks
        result["all_thresholds_passed"] = all_passed

        self.logger.info(
            f"阈值检查 | 全部达标={all_passed} | "
            + " | ".join(
                f"{k}={v['actual']:.4f}>={v['threshold']}"
                for k, v in threshold_checks.items()
            )
        )

        return result
