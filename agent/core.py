# -*- coding: utf-8 -*-
"""
Agent 任务循环主控
===================
SegmentationAgent 类编排完整的自主迭代优化流程：
    数据集校验 -> 训练 -> 评估 -> 调参决策 -> 是否继续迭代 -> 汇总实验结果
包含完整异常捕获，对训练报错、数据集异常做日志记录。
"""

import os
import sys
import copy
import traceback

import numpy as np
import torch

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.dataset_validator import DatasetValidator
from pipeline.trainer import UNetTrainer
from pipeline.evaluator import Evaluator
from pipeline.hyperparam_tuner import HyperparamTuner
from agent.task_planner import TaskPlanner, TerminationCondition
from utils.logger import ExperimentLogger
from utils.data_augment import SegmentationAugmentor
from llm.llm_advisor import LLMAdvisor


class SegmentationAgent:
    """图像分割自动调优 Agent

    职责:
        1. 校验数据集完整性
        2. 封装U-Net训练流程为可调度单元
        3. 自动读取IoU、Dice等分割评估指标
        4. 根据阈值自动调整学习率、数据增强策略
        5. 异常捕获与日志记录
        6. 实验指标自动汇总输出

    Args:
        config: 全局配置对象
        train_data: 训练数据(列表)
        val_data: 验证数据(列表)
        logger: 日志记录器
    """

    def __init__(self, config, train_data=None, val_data=None, logger=None):
        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        self.logger = logger or ExperimentLogger(
            log_file=getattr(config, "LOG_FILE", None),
            experiment_name="segmentation_agent",
        )

        # 从config提取参数
        self.num_classes = getattr(config, "NUM_CLASSES", 2)
        self.device = getattr(config, "DEVICE", "cpu")
        self.image_dir = getattr(config, "IMAGE_DIR", None)
        self.label_dir = getattr(config, "LABEL_DIR", None)
        self.results_dir = getattr(config, "RESULTS_DIR", "results")

        # 终止条件
        termination = TerminationCondition(
            iou_threshold=getattr(config, "IOU_THRESHOLD", 0.70),
            dice_threshold=getattr(config, "DICE_THRESHOLD", 0.75),
            pixel_acc_threshold=getattr(config, "PIXEL_ACC_THRESHOLD", 0.90),
            max_iterations=getattr(config, "MAX_ITERATIONS", 5),
            patience=getattr(config, "PATIENCE", 2),
            min_delta=getattr(config, "MIN_DELTA", 0.005),
        )

        # 初始化大模型顾问(若启用)
        self.use_llm = getattr(config, "USE_LLM_ADVISOR", False)
        self.llm_advisor = None
        if self.use_llm:
            self.llm_advisor = LLMAdvisor(
                config=getattr(config, "LLM_CONFIG", None),
                logger=self.logger,
                safety_bounds=getattr(config, "LLM_SAFETY_BOUNDS", None),
            )
            mode = "大模型智能" if not self.llm_advisor.is_local else "本地模拟(降级)"
            self.logger.info(
                f"[大模型顾问] 已启用，模式: {mode} "
                f"(provider={self.llm_advisor.provider}, "
                f"model={self.llm_advisor.model})"
            )

        # 初始化组件
        self.task_planner = TaskPlanner(
            termination=termination,
            logger=self.logger,
            llm_advisor=self.llm_advisor,  # 注入大模型顾问
        )
        self.tuner = HyperparamTuner(
            lr_decay_factor=getattr(config, "LR_DECAY_FACTOR", 0.5),
            lr_min=getattr(config, "LR_MIN", 1e-6),
            logger=self.logger,
            llm_advisor=self.llm_advisor,  # 注入大模型顾问
        )

        # 当前超参配置(深拷贝，避免嵌套 dict 被调参器原地修改)
        self.current_config = copy.deepcopy(
            getattr(config, "INITIAL_CONFIG", {})
        )
        self.warm_start = getattr(config, "WARM_START", True)
        self._model_state = None
        self._best_iou = -1.0

        # 阈值字典
        self.thresholds = {
            "iou": termination.iou_threshold,
            "dice": termination.dice_threshold,
            "pixel_accuracy": termination.pixel_acc_threshold,
        }

        # 实验记录
        self.experiment_records = []

        # 确保结果目录存在
        os.makedirs(self.results_dir, exist_ok=True)

    def run(self):
        """执行完整的自动调优流程

        主循环:
            1. 校验数据集(首轮)
            2. 训练模型
            3. 评估指标
            4. 调参决策
            5. 判断是否继续
            6. 汇总输出

        Returns:
            dict: 实验汇总报告
        """
        self.logger.info("=" * 60)
        self.logger.info("图像分割自动调优 Agent 启动")
        self.logger.info("=" * 60)

        try:
            # ==================== 1. 数据集校验 ====================
            if self.train_data is None and self.image_dir and self.label_dir:
                self.logger.info("[步骤1] 数据集校验")
                validator = DatasetValidator(
                    image_dir=self.image_dir,
                    label_dir=self.label_dir,
                    logger=self.logger,
                    supported_image_formats=getattr(
                        self.config, "SUPPORTED_IMAGE_FORMATS", None),
                    expected_size=getattr(self.config, "EXPECTED_IMAGE_SIZE", None),
                    label_value_range=getattr(
                        self.config, "LABEL_VALUE_RANGE", None),
                    num_classes=self.num_classes,
                )
                validation_result = validator.validate()
                if not validation_result["valid"]:
                    self.logger.error(
                        f"数据集校验未通过: {validation_result['issues']}"
                    )
                    return self._build_summary(
                        success=False,
                        reason="数据集校验未通过",
                        issues=validation_result["issues"],
                    )
                loaded = validator.load_pairs(validation_result["file_pairs"])
                split = max(1, int(len(loaded) * 0.8))
                self.train_data = loaded[:split]
                self.val_data = loaded[split:] or loaded[:1]
                self.logger.info(
                    f"磁盘数据集已加载: 训练{len(self.train_data)} / "
                    f"验证{len(self.val_data)}"
                )
            else:
                self.logger.info("[步骤1] 使用传入的数据(跳过文件系统校验)")

            # ==================== 2. 迭代优化循环 ====================
            iteration = 0
            final_metrics = {}

            while True:
                self.logger.info(f"\n{'='*40}")
                self.logger.info(f"迭代轮次 {iteration + 1}")
                self.logger.info(f"{'='*40}")

                # 获取本轮迭代步骤
                steps = self.task_planner.get_iteration_steps(iteration)
                self.logger.info(
                    f"本轮步骤: {' -> '.join(s.step_name for s in steps)}"
                )

                try:
                    # --- 步骤2a: 训练 ---
                    self.logger.info(f"[训练] 使用配置: lr="
                                     f"{self.current_config.get('learning_rate')}, "
                                     f"epochs={self.current_config.get('num_epochs')}")

                    augmentor = SegmentationAugmentor(
                        aug_config=self.current_config.get("augmentation", {}),
                        prob=0.5,
                    )

                    trainer = UNetTrainer(
                        config=self.current_config,
                        num_classes=self.num_classes,
                        in_channels=3,
                        device=self.device,
                        logger=self.logger,
                        init_state=self._model_state if self.warm_start else None,
                        base_features=getattr(self.config, "INITIAL_FEATURES", 64),
                        grad_clip_norm=getattr(self.config, "GRAD_CLIP_NORM", 1.0),
                    )

                    train_result = trainer.train(
                        train_data=self.train_data,
                        val_data=self.val_data,
                        augmentor=augmentor,
                    )

                    model = train_result["model"]
                    self._model_state = train_result.get("model_state")
                    final_metrics = dict(train_result.get("best_metrics") or {})
                    final_metrics["loss"] = train_result.get("final_loss", 0)

                    # --- 步骤2b: 评估(权重已恢复为最佳 checkpoint) ---
                    self.logger.info("[评估] 在验证集上计算指标")
                    if self.val_data:
                        evaluator = Evaluator(
                            model=model,
                            device=self.device,
                            num_classes=self.num_classes,
                            logger=self.logger,
                        )
                        eval_result = evaluator.evaluate_and_compare(
                            self.val_data, self.thresholds
                        )
                        final_metrics = dict(eval_result["metrics"])
                        final_metrics["loss"] = train_result.get("final_loss", 0)

                    current_iou = final_metrics.get("iou", 0)
                    if current_iou > self._best_iou and self._model_state is not None:
                        self._best_iou = current_iou
                        ckpt_path = os.path.join(self.results_dir, "best_model.pth")
                        torch.save(self._model_state, ckpt_path)
                        self.logger.info(f"[训练] 最佳模型已保存: {ckpt_path}")

                    # 记录指标
                    self.logger.log_metrics(iteration + 1, final_metrics, phase="val")

                    # 保存实验记录
                    self.experiment_records.append({
                        "iteration": iteration + 1,
                        "config": copy.deepcopy(self.current_config),
                        "metrics": copy.deepcopy(final_metrics)
                            if isinstance(final_metrics, dict) else {},
                        "loss": train_result.get("final_loss", 0),
                    })

                    # --- 步骤2c: 调参决策 ---
                    self.logger.info("[调参] 根据指标调整超参")
                    new_config = self.tuner.tune(
                        current_config=self.current_config,
                        metrics=final_metrics,
                        thresholds=self.thresholds,
                        iteration=iteration + 1,
                        max_iter=self.task_planner.termination.max_iterations,
                        history=self.experiment_records,
                    )
                    self.current_config = new_config

                    # 将大模型的继续/终止建议注入 task_planner
                    if self.tuner.last_decision is not None:
                        self.task_planner.set_llm_decision(
                            self.tuner.last_decision.get("should_continue", True)
                        )

                    # --- 步骤2d: 终止判断 ---
                    should_stop, reason = self.task_planner.check_termination(
                        final_metrics, iteration
                    )

                    if should_stop:
                        self.logger.info(f"迭代终止: {reason}")
                        break

                    iteration += 1

                except Exception as e:
                    self.logger.log_exception(e, context=f"迭代{iteration+1}训练过程")
                    iteration += 1
                    if iteration >= self.task_planner.termination.max_iterations:
                        self.logger.error("达到最大迭代次数，终止")
                        break
                    continue

            # ==================== 3. 汇总输出 ====================
            summary = self._build_summary(
                success=True,
                final_metrics=final_metrics,
                iterations=iteration + 1,
            )
            self.logger.log_experiment_summary(summary)
            self._save_summary_report(summary)

            return summary

        except Exception as e:
            self.logger.log_exception(e, context="Agent主循环")
            return self._build_summary(
                success=False,
                reason=f"Agent异常终止: {e}",
            )

    def _build_summary(self, success=True, reason="", **kwargs):
        """构建实验汇总报告"""
        summary = {
            "success": success,
            "reason": reason,
            "total_iterations": len(self.experiment_records),
            "thresholds": self.thresholds,
            "final_config": self.current_config,
        }

        # 添加指标历史
        metrics_history = self.logger.get_metrics_history()
        if metrics_history:
            best = self.logger.get_best_metrics("iou")
            summary["best_metrics"] = best
            summary["metrics_history"] = metrics_history
            summary["final_metrics"] = metrics_history[-1]

        # 添加传入的额外信息
        summary.update(kwargs)

        return summary

    def _save_summary_report(self, summary):
        """保存汇总报告到文件"""
        report_path = os.path.join(self.results_dir, "summary_report.txt")
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write("图像分割自动调优Agent - 实验汇总报告\n")
                f.write("=" * 60 + "\n\n")

                f.write(f"实验状态: {'成功' if summary['success'] else '失败'}\n")
                f.write(f"总迭代轮数: {summary['total_iterations']}\n")
                f.write(f"终止原因: {summary.get('reason', 'N/A')}\n\n")

                f.write("评估阈值:\n")
                for k, v in summary.get("thresholds", {}).items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")

                if "final_metrics" in summary:
                    fm = summary["final_metrics"]
                    f.write("最终指标:\n")
                    f.write(f"  IoU: {fm.get('iou', 0):.4f}\n")
                    f.write(f"  Dice: {fm.get('dice', 0):.4f}\n")
                    f.write(f"  Pixel Accuracy: {fm.get('pixel_accuracy', 0):.4f}\n\n")

                if "best_metrics" in summary and summary["best_metrics"]:
                    bm = summary["best_metrics"]
                    f.write("最佳指标记录:\n")
                    f.write(f"  迭代轮次: {bm.get('iteration', 'N/A')}\n")
                    f.write(f"  IoU: {bm.get('iou', 0):.4f}\n")
                    f.write(f"  Dice: {bm.get('dice', 0):.4f}\n")
                    f.write(f"  PA: {bm.get('pixel_accuracy', 0):.4f}\n\n")

                f.write("最终超参配置:\n")
                for k, v in summary.get("final_config", {}).items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")

                f.write("迭代指标历史:\n")
                for record in self.experiment_records:
                    metrics = record.get("metrics", {})
                    f.write(
                        f"  迭代{record['iteration']}: "
                        f"IoU={metrics.get('iou', 0):.4f} "
                        f"Dice={metrics.get('dice', 0):.4f} "
                        f"PA={metrics.get('pixel_accuracy', 0):.4f} "
                        f"loss={record.get('loss', 0):.4f}\n"
                    )

            # ============ 大模型生成分析报告 ============
            if self.llm_advisor is not None:
                try:
                    self.logger.info("[大模型顾问] 生成实验分析报告...")
                    llm_report = self.llm_advisor.generate_report(
                        summary=summary,
                        history=self.experiment_records,
                    )
                    with open(report_path, "a", encoding="utf-8") as f:
                        f.write("\n" + "=" * 60 + "\n")
                        f.write("大模型实验分析报告\n")
                        f.write("=" * 60 + "\n\n")
                        f.write(llm_report + "\n")
                    self.logger.info("[大模型顾问] 分析报告已追加到汇总文件")
                    summary["llm_report"] = llm_report
                except Exception as e:
                    self.logger.warning(f"大模型报告生成失败: {e}")

            self.logger.info(f"汇总报告已保存到: {report_path}")
        except Exception as e:
            self.logger.log_exception(e, context="保存汇总报告")

    def run_comparison_experiments(self):
        """运行对照实验

        在不同超参组合下进行对照实验，对比分析结果。

        Returns:
            list: 各实验的结果
        """
        self.logger.info("=" * 60)
        self.logger.info("开始对照实验")
        self.logger.info("=" * 60)

        plans = self.task_planner.generate_experiment_plans(self.current_config)
        results = []

        for plan in plans:
            self.logger.info(f"\n[对照实验] {plan.experiment_name}: {plan.description}")

            try:
                # 临时使用实验配置
                original_config = self.current_config
                self.current_config = plan.config.copy()

                # 简化版训练(单轮)
                trainer = UNetTrainer(
                    config=self.current_config,
                    num_classes=self.num_classes,
                    in_channels=3,
                    device=self.device,
                    logger=self.logger,
                    base_features=getattr(self.config, "INITIAL_FEATURES", 64),
                    grad_clip_norm=getattr(self.config, "GRAD_CLIP_NORM", 1.0),
                )
                augmentor = SegmentationAugmentor(
                    aug_config=self.current_config.get("augmentation", {}),
                    prob=0.5,
                )
                train_result = trainer.train(
                    train_data=self.train_data,
                    val_data=self.val_data,
                    augmentor=augmentor,
                )

                result = {
                    "experiment_name": plan.experiment_name,
                    "description": plan.description,
                    "metrics": train_result["best_metrics"],
                    "loss": train_result.get("final_loss", 0),
                }
                results.append(result)

                self.logger.info(
                    f"[对照实验] {plan.experiment_name} 完成 | "
                    f"IoU={result['metrics'].get('iou', 0):.4f}"
                )

                # 恢复原配置
                self.current_config = original_config

            except Exception as e:
                self.logger.log_exception(
                    e, context=f"对照实验{plan.experiment_name}"
                )
                results.append({
                    "experiment_name": plan.experiment_name,
                    "description": plan.description,
                    "error": str(e),
                })

        # 输出对照实验汇总
        self.logger.info("\n" + "=" * 60)
        self.logger.info("对照实验汇总")
        self.logger.info("=" * 60)
        for r in results:
            if "error" in r:
                self.logger.info(f"  {r['experiment_name']}: 失败 - {r['error']}")
            else:
                m = r["metrics"]
                self.logger.info(
                    f"  {r['experiment_name']}: "
                    f"IoU={m.get('iou', 0):.4f} "
                    f"Dice={m.get('dice', 0):.4f}"
                )

        return results
