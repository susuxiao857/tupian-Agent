# -*- coding: utf-8 -*-
"""
任务执行流程规划
==================
规划 Agent 的迭代流程：定义迭代步骤、终止条件、对照实验配置。
是 Agent 的"大脑"，决定每一步做什么、什么时候停。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any

from utils.logger import ExperimentLogger


@dataclass
class IterationStep:
    """迭代步骤定义"""
    step_name: str          # 步骤名称
    description: str        # 步骤描述
    action: str             # 执行动作标识
    required: bool = True   # 是否必须执行


@dataclass
class TerminationCondition:
    """终止条件"""
    iou_threshold: float = 0.70       # IoU达标阈值
    dice_threshold: float = 0.75      # Dice达标阈值
    pixel_acc_threshold: float = 0.90 # 像素准确率达标阈值
    max_iterations: int = 5           # 最大迭代轮数
    patience: int = 2                 # 平台期观察窗口(轮)
    min_delta: float = 0.005          # IoU 最小有效提升


@dataclass
class ExperimentPlan:
    """对照实验配置"""
    experiment_name: str              # 实验名称
    config: Dict[str, Any] = field(default_factory=dict)  # 超参配置
    description: str = ""              # 实验描述


class TaskPlanner:
    """任务执行流程规划器

    职责:
        1. 定义每一轮迭代的执行步骤序列
        2. 管理终止条件(指标达标或达到最大迭代次数)
        3. 生成对照实验配置(不同超参组合)

    大模型可选接入：当注入 llm_advisor 且 llm_advisor 在调参时已
    给出 should_continue 决策，check_termination 会综合大模型建议
    与规则阈值双重判断。

    Args:
        termination: 终止条件
        logger: 日志记录器
        llm_advisor: 大模型顾问(可选，用于辅助终止决策)
    """

    def __init__(self, termination=None, logger=None, llm_advisor=None):
        self.termination = termination or TerminationCondition()
        self.logger = logger or ExperimentLogger()
        self.llm_advisor = llm_advisor
        self.iteration_count = 0
        self._last_llm_should_continue = None
        self._iou_history = []

    def get_iteration_steps(self, iteration):
        """获取一轮迭代的执行步骤序列

        每轮迭代包含以下步骤:
        1. 数据集校验(首轮)
        2. 训练模型
        3. 评估指标
        4. 调参决策
        5. 判断是否继续

        Args:
            iteration: 当前迭代轮次(0-based)

        Returns:
            List[IterationStep]: 步骤序列
        """
        steps = []

        # 首轮迭代需要校验数据集
        if iteration == 0:
            steps.append(IterationStep(
                step_name="dataset_validation",
                description="校验数据集：检查图像/标签格式、尺寸、类别完整性",
                action="validate_dataset",
                required=True,
            ))

        # 训练
        steps.append(IterationStep(
            step_name="training",
            description=f"迭代{iteration+1}：使用当前超参配置训练U-Net模型",
            action="train_model",
            required=True,
        ))

        # 评估
        steps.append(IterationStep(
            step_name="evaluation",
            description=f"迭代{iteration+1}：在验证集上计算IoU、Dice、像素准确率",
            action="evaluate_model",
            required=True,
        ))

        # 调参
        steps.append(IterationStep(
            step_name="hyperparameter_tuning",
            description=f"迭代{iteration+1}：根据指标与阈值对比，调整学习率和数据增强",
            action="tune_hyperparams",
            required=True,
        ))

        # 终止判断
        steps.append(IterationStep(
            step_name="termination_check",
            description=f"迭代{iteration+1}：检查是否达标或达到最大迭代次数",
            action="check_termination",
            required=True,
        ))

        return steps

    def set_llm_decision(self, should_continue):
        """注入最近一次大模型的继续/终止建议(由 Agent 在调参后调用)

        Args:
            should_continue: 大模型建议是否继续迭代
        """
        self._last_llm_should_continue = should_continue

    def check_termination(self, metrics, iteration):
        """检查是否满足终止条件

        终止逻辑(优先级从高到低):
            1. 指标达标(IoU+Dice 同时达标) -> 终止
            2. 达到最大迭代次数 -> 终止
            3. 指标平台期(连续多轮无明显提升) -> 终止
            4. 大模型建议终止(且未达标) -> 终止
            5. 否则继续

        Args:
            metrics: 当前评估指标
            iteration: 当前迭代轮次

        Returns:
            tuple: (should_stop: bool, reason: str)
        """
        # 1. 检查指标是否达标
        iou = metrics.get("iou", 0.0)
        dice = metrics.get("dice", 0.0)
        pa = metrics.get("pixel_accuracy", 0.0)

        thresholds_met = []
        if iou >= self.termination.iou_threshold:
            thresholds_met.append(f"IoU={iou:.4f}>={self.termination.iou_threshold}")
        if dice >= self.termination.dice_threshold:
            thresholds_met.append(f"Dice={dice:.4f}>={self.termination.dice_threshold}")
        if pa >= self.termination.pixel_acc_threshold:
            thresholds_met.append(f"PA={pa:.4f}>={self.termination.pixel_acc_threshold}")

        # IoU和Dice同时达标则停止
        if (iou >= self.termination.iou_threshold and
                dice >= self.termination.dice_threshold):
            reason = f"指标达标，停止迭代: {' | '.join(thresholds_met)}"
            self.logger.info(f"[终止判断] {reason}")
            return True, reason

        # 2. 检查是否达到最大迭代次数
        if iteration + 1 >= self.termination.max_iterations:
            reason = (f"达到最大迭代次数({self.termination.max_iterations})，停止迭代。"
                      f"最终指标: IoU={iou:.4f}, Dice={dice:.4f}")
            self.logger.info(f"[终止判断] {reason}")
            return True, reason

        # 3. 指标平台期
        self._iou_history.append(iou)
        patience = getattr(self.termination, "patience", 2)
        min_delta = getattr(self.termination, "min_delta", 0.005)
        if patience > 0 and len(self._iou_history) >= patience:
            window = self._iou_history[-patience:]
            if (max(window) - min(window)) < min_delta:
                reason = (
                    f"指标平台期(近{patience}轮 IoU 波动 < {min_delta})，停止迭代。"
                    f"最终指标: IoU={iou:.4f}, Dice={dice:.4f}"
                )
                self.logger.info(f"[终止判断] {reason}")
                return True, reason

        # 4. 大模型建议终止(优先级低于规则达标，高于"继续")
        if (self.llm_advisor is not None
                and self._last_llm_should_continue is False):
            reason = (f"大模型建议停止迭代(综合判断指标已无提升空间)。"
                      f"最终指标: IoU={iou:.4f}, Dice={dice:.4f}")
            self.logger.info(f"[终止判断] {reason}")
            return True, reason

        # 4. 继续迭代
        reason = (f"指标未达标，继续迭代。"
                  f"IoU={iou:.4f}/{self.termination.iou_threshold}, "
                  f"Dice={dice:.4f}/{self.termination.dice_threshold}")
        self.logger.info(f"[终止判断] {reason}")
        return False, reason

    def generate_experiment_plans(self, base_config):
        """生成对照实验配置

        在不同超参组合下进行对照实验，用于对比分析。

        Args:
            base_config: 基础超参配置

        Returns:
            List[ExperimentPlan]: 对照实验配置列表
        """
        import copy

        plans = []

        # 实验1：基础配置(无数据增强)
        config_no_aug = copy.deepcopy(base_config)
        config_no_aug["augmentation"] = {
            "horizontal_flip": False,
            "vertical_flip": False,
            "random_rotation": False,
            "elastic_deform": False,
        }
        plans.append(ExperimentPlan(
            experiment_name="baseline_no_aug",
            config=config_no_aug,
            description="基线实验：无数据增强",
        ))

        # 实验2：基础配置(有数据增强)
        config_with_aug = copy.deepcopy(base_config)
        config_with_aug["augmentation"] = {
            "horizontal_flip": True,
            "vertical_flip": True,
            "random_rotation": False,
            "elastic_deform": False,
        }
        plans.append(ExperimentPlan(
            experiment_name="baseline_with_flip",
            config=config_with_aug,
            description="对照实验：水平+垂直翻转",
        ))

        # 实验3：高学习率
        config_high_lr = copy.deepcopy(base_config)
        config_high_lr["learning_rate"] = 5e-3
        plans.append(ExperimentPlan(
            experiment_name="high_lr",
            config=config_high_lr,
            description="对照实验：高学习率5e-3",
        ))

        # 实验4：低学习率
        config_low_lr = copy.deepcopy(base_config)
        config_low_lr["learning_rate"] = 5e-4
        plans.append(ExperimentPlan(
            experiment_name="low_lr",
            config=config_low_lr,
            description="对照实验：低学习率5e-4",
        ))

        self.logger.info(f"生成对照实验配置 {len(plans)} 组")
        return plans
