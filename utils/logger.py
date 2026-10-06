# -*- coding: utf-8 -*-
"""
实验日志记录器
==================
实现 ExperimentLogger 类，记录训练日志、异常、指标变化，支持写入文件。
用于训练报错、数据集异常等情况的日志记录与追溯。
"""

import logging
import os
import sys
from datetime import datetime


class ExperimentLogger:
    """实验日志记录器

    功能:
        1. 记录训练过程日志(INFO/DEBUG级别)
        2. 捕获并记录异常(ERROR级别)，含堆栈信息
        3. 记录指标变化(每次迭代的IoU/Dice等)
        4. 输出到控制台和文件

    Args:
        log_file: 日志文件路径
        experiment_name: 实验名称
        level: 日志级别
        to_console: 是否输出到控制台
    """

    def __init__(self, log_file=None, experiment_name="segmentation_agent",
                 level=logging.INFO, to_console=True):
        self.experiment_name = experiment_name
        self.log_file = log_file
        self._metrics_history = []  # 指标变化历史

        # 创建日志器
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(level)
        # 避免重复添加handler
        self.logger.handlers = []
        self.logger.propagate = False

        # 日志格式
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 文件handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # 控制台handler
        if to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        self.info(f"实验日志器初始化完成，实验名称: {experiment_name}")

    # ==================== 基本日志方法 ====================
    def info(self, message):
        """记录INFO级别日志"""
        self.logger.info(message)

    def debug(self, message):
        """记录DEBUG级别日志"""
        self.logger.debug(message)

    def warning(self, message):
        """记录WARNING级别日志"""
        self.logger.warning(message)

    def error(self, message, exc_info=None):
        """记录ERROR级别日志，可附带异常堆栈"""
        if exc_info is not None:
            self.logger.error(message, exc_info=exc_info)
        else:
            self.logger.error(message)

    # ==================== 异常记录 ====================
    def log_exception(self, exception, context=""):
        """记录异常信息

        Args:
            exception: 异常对象
            context: 异常发生的上下文描述
        """
        import traceback
        error_msg = f"异常发生 - {context}" if context else "异常发生"
        self.error(error_msg, exc_info=exception)
        tb_str = traceback.format_exc()
        self.error(f"异常堆栈: {tb_str}")

    def log_dataset_anomaly(self, anomaly_desc, file_path=None):
        """记录数据集异常

        Args:
            anomaly_desc: 异常描述
            file_path: 相关文件路径
        """
        prefix = f"[数据集异常] 文件: {file_path} | " if file_path else "[数据集异常] "
        self.warning(prefix + anomaly_desc)

    # ==================== 指标记录 ====================
    def log_metrics(self, iteration, metrics, phase="val"):
        """记录指标并保存到历史

        Args:
            iteration: 迭代轮次
            metrics: 指标字典(包含iou, dice, pixel_accuracy等)
            phase: 阶段(train/val)
        """
        record = {
            "iteration": iteration,
            "phase": phase,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "iou": metrics.get("iou", 0.0),
            "dice": metrics.get("dice", 0.0),
            "pixel_accuracy": metrics.get("pixel_accuracy", 0.0),
        }
        self._metrics_history.append(record)
        self.info(
            f"[指标记录] 迭代{iteration} {phase}阶段 | "
            f"IoU={record['iou']:.4f} Dice={record['dice']:.4f} "
            f"PA={record['pixel_accuracy']:.4f}"
        )

    def get_metrics_history(self):
        """获取指标变化历史"""
        return self._metrics_history

    def get_best_metrics(self, metric_name="iou"):
        """获取最佳指标记录

        Args:
            metric_name: 指标名称(iou/dice/pixel_accuracy)

        Returns:
            dict: 最佳指标记录，若无则返回None
        """
        if not self._metrics_history:
            return None
        return max(self._metrics_history, key=lambda x: x.get(metric_name, 0))

    # ==================== 实验汇总 ====================
    def log_experiment_summary(self, summary):
        """记录实验汇总报告

        Args:
            summary: 汇总信息字典
        """
        self.info("=" * 60)
        self.info("实验汇总报告")
        self.info("=" * 60)
        for key, value in summary.items():
            self.info(f"  {key}: {value}")
        self.info("=" * 60)
