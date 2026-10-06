# -*- coding: utf-8 -*-
"""
入口脚本
==================
命令行入口：加载数据集配置 -> 启动Agent -> 输出实验汇总报告。
"""

import argparse
import os
import sys

import numpy as np

# 将项目根目录加入路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config
from agent.core import SegmentationAgent
from utils.logger import ExperimentLogger
from utils.repro import set_seed


def generate_synthetic_dataset(num_samples=20, img_size=64, seed=42):
    """生成合成数据集(无需真实数据即可体验Agent流程)

    生成包含随机圆形的合成图像和对应的二值分割标签。

    Args:
        num_samples: 样本数量
        img_size: 图像尺寸
        seed: 随机种子

    Returns:
        list: [(image, label), ...]
    """
    rng = np.random.RandomState(seed)
    data = []

    for i in range(num_samples):
        image = np.ones((img_size, img_size, 3), dtype=np.uint8) * 128
        label = np.zeros((img_size, img_size), dtype=np.uint8)

        num_circles = rng.randint(1, 4)
        for _ in range(num_circles):
            cx = rng.randint(img_size // 4, 3 * img_size // 4)
            cy = rng.randint(img_size // 4, 3 * img_size // 4)
            radius = rng.randint(5, img_size // 6)

            y, x = np.ogrid[:img_size, :img_size]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2

            color = rng.randint(50, 255, size=3)
            image[mask] = color
            label[mask] = 1

        noise = rng.randint(-20, 20, size=image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        data.append((image, label))

    return data


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description="面向图像分割任务的自主迭代优化Agent"
    )
    parser.add_argument(
        "--mode", type=str, default="auto",
        choices=["auto", "compare", "both"],
        help="运行模式: auto=自动调优, compare=对照实验, both=两者都执行",
    )
    parser.add_argument(
        "--synthetic", action="store_true", default=True,
        help="使用合成数据(无需真实数据集，默认开启)",
    )
    parser.add_argument(
        "--real-data", action="store_true",
        help="从 data/images 与 data/labels 加载真实数据(覆盖 --synthetic)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=20,
        help="合成数据样本数量",
    )
    parser.add_argument(
        "--img_size", type=int, default=64,
        help="合成图像尺寸",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=None,
        help="覆盖配置中的最大迭代次数",
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda"],
        help="覆盖计算设备",
    )
    args = parser.parse_args()

    set_seed(getattr(config, "RANDOM_SEED", 42))
    if args.max_iterations is not None:
        config.MAX_ITERATIONS = args.max_iterations
    if args.device:
        config.DEVICE = args.device

    print("=" * 60)
    print("面向图像分割任务的自主迭代优化Agent")
    print("=" * 60)

    # 显示大模型接入状态
    if getattr(config, "USE_LLM_ADVISOR", False):
        provider = config.LLM_CONFIG.get("provider", "local")
        model = config.LLM_CONFIG.get("model", "local-simulator")
        api_key = config.LLM_CONFIG.get("api_key", "")
        if provider == "local" or not api_key:
            print(f"大模型顾问：本地模拟模式(无需 API Key，规则模拟决策)")
        else:
            print(f"大模型顾问：{provider} / {model} (智能决策)")
            print("  能力：诊断训练状态 + 自主调参 + 终止决策 + 分析报告")
    else:
        print("大模型顾问：未启用(规则驱动调参)")
    print("=" * 60)

    # 创建日志器
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    logger = ExperimentLogger(
        log_file=config.LOG_FILE,
        experiment_name="segmentation_agent",
    )

    logger.info("配置加载完成:")
    logger.info(f"  数据集目录: {config.DATA_DIR}")
    logger.info(f"  结果目录: {config.RESULTS_DIR}")
    logger.info(f"  初始学习率: {config.INITIAL_CONFIG['learning_rate']}")
    logger.info(f"  Batch大小: {config.INITIAL_CONFIG['batch_size']}")
    logger.info(f"  训练轮数: {config.INITIAL_CONFIG['num_epochs']}")
    logger.info(f"  IoU阈值: {config.IOU_THRESHOLD}")
    logger.info(f"  Dice阈值: {config.DICE_THRESHOLD}")
    logger.info(f"  最大迭代次数: {config.MAX_ITERATIONS}")
    logger.info(f"  数据增强: {config.INITIAL_CONFIG['augmentation']}")
    logger.info(f"  设备: {config.DEVICE}")

    # 准备数据
    logger.info("准备数据...")
    train_data = None
    val_data = None
    if not args.real_data:
        train_data = generate_synthetic_dataset(
            num_samples=args.num_samples,
            img_size=args.img_size,
            seed=42,
        )
        val_data = generate_synthetic_dataset(
            num_samples=max(4, args.num_samples // 3),
            img_size=args.img_size,
            seed=99,
        )
        logger.info(f"训练数据: {len(train_data)}个样本(合成)")
        logger.info(f"验证数据: {len(val_data)}个样本(合成)")
    else:
        logger.info("使用真实数据集(由 Agent 校验并加载磁盘文件)")

    # 创建Agent
    agent = SegmentationAgent(
        config=config,
        train_data=train_data,
        val_data=val_data,
        logger=logger,
    )

    # 执行
    if args.mode in ("auto", "both"):
        logger.info("\n" + "=" * 60)
        logger.info("运行自动调优流程")
        logger.info("=" * 60)
        summary = agent.run()
        print_summary(summary)

    if args.mode in ("compare", "both"):
        logger.info("\n" + "=" * 60)
        logger.info("运行对照实验")
        logger.info("=" * 60)
        results = agent.run_comparison_experiments()
        print_comparison_results(results)

    logger.info("\n完成！结果已保存到: " + config.RESULTS_DIR)


def print_summary(summary):
    """打印实验汇总"""
    print("\n" + "=" * 60)
    print("实验汇总报告")
    print("=" * 60)
    print(f"状态: {'成功' if summary['success'] else '失败'}")
    print(f"终止原因: {summary.get('reason', 'N/A')}")
    print(f"总迭代轮数: {summary['total_iterations']}")
    print()

    if "final_metrics" in summary:
        fm = summary["final_metrics"]
        print("最终指标:")
        print(f"  IoU: {fm.get('iou', 0):.4f}")
        print(f"  Dice: {fm.get('dice', 0):.4f}")
        print(f"  Pixel Accuracy: {fm.get('pixel_accuracy', 0):.4f}")
        print()

    if "best_metrics" in summary and summary["best_metrics"]:
        bm = summary["best_metrics"]
        print("最佳指标:")
        print(f"  迭代轮次: {bm.get('iteration', 'N/A')}")
        print(f"  IoU: {bm.get('iou', 0):.4f}")
        print(f"  Dice: {bm.get('dice', 0):.4f}")
        print()

    print("最终超参配置:")
    for k, v in summary.get("final_config", {}).items():
        print(f"  {k}: {v}")
    print()

    # 展示大模型生成的分析报告
    if "llm_report" in summary and summary["llm_report"]:
        print("-" * 60)
        print("大模型实验分析报告:")
        print("-" * 60)
        print(summary["llm_report"])
        print()
    print("=" * 60)


def print_comparison_results(results):
    """打印对照实验结果"""
    print("\n" + "=" * 60)
    print("对照实验结果")
    print("=" * 60)
    for r in results:
        if "error" in r:
            print(f"  {r['experiment_name']}: 失败 - {r['error']}")
        else:
            m = r["metrics"]
            print(
                f"  {r['experiment_name']}: "
                f"IoU={m.get('iou', 0):.4f} "
                f"Dice={m.get('dice', 0):.4f} "
                f"PA={m.get('pixel_accuracy', 0):.4f}"
            )
    print("=" * 60)


if __name__ == "__main__":
    main()
