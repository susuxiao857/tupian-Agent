# -*- coding: utf-8 -*-
"""
超参数调优模块
==================
根据当前评估指标与阈值的对比，自动调整学习率(衰减)和数据增强策略(增强/减弱)，
返回新的超参配置。
"""

import copy

from utils.logger import ExperimentLogger


class HyperparamTuner:
    """超参数自动调优器

    调优策略:
        1. 学习率衰减：每轮迭代后，如果未达标则将学习率乘以衰减因子
        2. 数据增强策略调整：
           - 指标较低时增强数据增强强度(防止过拟合)
           - 指标较高但未达标时减弱数据增强(更关注训练拟合)
           - 指标已达标则保持不变
        3. 批大小/训练轮数微调

    Args:
        lr_decay_factor: 学习率衰减因子
        lr_min: 学习率下限
        logger: 日志记录器
    """

    def __init__(self, lr_decay_factor=0.5, lr_min=1e-6, logger=None,
                 llm_advisor=None):
        self.lr_decay_factor = lr_decay_factor
        self.lr_min = lr_min
        self.logger = logger or ExperimentLogger()
        self.llm_advisor = llm_advisor  # 大模型顾问(可选)
        self.tuning_history = []
        # 缓存最近一次大模型完整决策(供 Agent 读取 should_continue)
        self.last_decision = None

    def tune(self, current_config, metrics, thresholds, iteration,
             max_iter=None, history=None):
        """根据当前指标自动调整超参配置

        若注入了大模型顾问，则由大模型决策调整方案；否则使用规则策略。
        大模型决策失败时自动降级规则模式。

        Args:
            current_config: 当前超参配置字典
            metrics: 当前评估指标字典
            thresholds: 目标阈值字典 {"iou": 0.7, "dice": 0.75}
            iteration: 当前迭代轮次
            max_iter: 最大迭代轮数(大模型决策时使用)
            history: 历史迭代记录(大模型决策时使用)

        Returns:
            dict: 调整后的新超参配置
        """
        # ============ 大模型智能决策分支 ============
        if self.llm_advisor is not None:
            try:
                decision = self.llm_advisor.advise_tuning(
                    metrics=metrics,
                    current_config=current_config,
                    history=history or [],
                    thresholds=thresholds,
                    iteration=iteration,
                    max_iter=max_iter or 5,
                )
                new_config = decision.get("new_config", current_config)
                self.last_decision = decision  # 缓存完整决策
                # 记录调优历史
                self.tuning_history.append({
                    "iteration": iteration,
                    "old_config": current_config,
                    "new_config": new_config,
                    "metrics": metrics,
                    "thresholds": thresholds,
                    "adjustments": [decision.get("diagnosis", ""),
                                    decision.get("reasoning", ""),
                                    decision.get("suggestions", "")],
                    "source": decision.get("source", "llm"),
                })
                self.logger.info(f"[调参] 迭代{iteration} 大模型决策:")
                self.logger.info(f"  诊断: {decision.get('diagnosis','')}")
                self.logger.info(f"  依据: {decision.get('reasoning','')}")
                self.logger.info(f"  建议: {decision.get('suggestions','')}")
                self.logger.info(f"  决策来源: {decision.get('source','llm')}")
                return new_config
            except Exception as e:
                self.logger.warning(
                    f"大模型调参决策异常，降级规则模式: {e}"
                )
                # 继续走规则策略

        # ============ 规则策略分支(原逻辑) ============
        new_config = copy.deepcopy(current_config)
        adjustments = []

        # --- 1. 学习率调整 ---
        current_lr = current_config.get("learning_rate", 1e-3)

        # 判断是否达标
        iou = metrics.get("iou", 0.0)
        iou_threshold = thresholds.get("iou", 0.7)
        lr_ratio = iou / iou_threshold if iou_threshold > 0 else 1.0

        if iou >= iou_threshold:
            # 已达标，轻微衰减学习率(精调)
            new_lr = max(current_lr * (self.lr_decay_factor ** 0.5), self.lr_min)
            new_config["learning_rate"] = new_lr
            adjustments.append(f"学习率精调: {current_lr:.2e} -> {new_lr:.2e}(已达标)")
        else:
            # 未达标，正常衰减学习率
            if lr_ratio < 0.5:
                # 指标远低于阈值：略降学习率并增加训练轮数(欠拟合更常见)
                new_lr = max(current_lr * self.lr_decay_factor, self.lr_min)
                adjustments.append(
                    f"学习率衰减: {current_lr:.2e} -> {new_lr:.2e}(指标偏低)"
                )
                current_epochs = current_config.get("num_epochs", 5)
                new_config["num_epochs"] = min(current_epochs + 2, 20)
                adjustments.append(
                    f"增加训练轮数: {current_epochs} -> {new_config['num_epochs']}(欠拟合)"
                )
            else:
                # 指标接近阈值，正常衰减
                new_lr = max(current_lr * self.lr_decay_factor, self.lr_min)
                adjustments.append(f"学习率衰减: {current_lr:.2e} -> {new_lr:.2e}")

        new_config["learning_rate"] = new_lr

        # --- 2. 数据增强策略调整 ---
        aug_config = new_config.setdefault("augmentation", {})
        dice = metrics.get("dice", 0.0)
        dice_threshold = thresholds.get("dice", 0.75)

        if iou >= iou_threshold and dice >= dice_threshold:
            # 已达标，保持增强策略不变
            adjustments.append("数据增强策略保持不变(已达标)")
        elif iou < 0.5 * iou_threshold:
            # 指标很低，可能欠拟合，减弱数据增强让模型更好地拟合训练数据
            for key in ["random_rotation", "elastic_deform"]:
                if aug_config.get(key, False):
                    aug_config[key] = False
                    adjustments.append(f"关闭{key}(指标偏低，减少正则化)")
        else:
            # 指标中等，可能过拟合，增强数据增强强度
            if not aug_config.get("horizontal_flip", False):
                aug_config["horizontal_flip"] = True
                adjustments.append("开启水平翻转(增强泛化)")
            if iou > 0.7 * iou_threshold and not aug_config.get("vertical_flip", False):
                aug_config["vertical_flip"] = True
                adjustments.append("开启垂直翻转(增强泛化)")

        new_config["augmentation"] = aug_config

        # --- 3. 训练轮数调整 ---
        # 指标接近达标时增加训练轮数
        if 0.8 < lr_ratio < 1.0:
            current_epochs = current_config.get("num_epochs", 5)
            new_config["num_epochs"] = min(current_epochs + 2, 20)
            adjustments.append(
                f"增加训练轮数: {current_epochs} -> {new_config['num_epochs']}(接近达标)"
            )

        # --- 记录调优历史 ---
        tuning_record = {
            "iteration": iteration,
            "old_config": current_config,
            "new_config": new_config,
            "metrics": metrics,
            "thresholds": thresholds,
            "adjustments": adjustments,
        }
        self.tuning_history.append(tuning_record)

        # 日志输出
        self.logger.info(f"[调参] 迭代{iteration} 超参数调整:")
        for adj in adjustments:
            self.logger.info(f"  - {adj}")

        return new_config

    def get_tuning_history(self):
        """获取调优历史"""
        return self.tuning_history

    def get_config_diff(self, old_config, new_config):
        """对比两个配置的差异

        Returns:
            dict: 变化的参数
        """
        diff = {}
        for key in new_config:
            if key not in old_config or old_config[key] != new_config[key]:
                diff[key] = {
                    "old": old_config.get(key),
                    "new": new_config[key],
                }
        return diff
