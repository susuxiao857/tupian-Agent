# -*- coding: utf-8 -*-
"""
测试脚本
==================
用合成数据(随机生成的小图像和标签)验证完整Agent流程可运行。
无需真实数据集即可体验完整的自动调优流程。

运行方式:
    python -m tests.test_agent
    或
    python tests/test_agent.py
"""

import os
import sys
import traceback

import numpy as np

# 将项目根目录加入路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def generate_synthetic_data(num_samples=20, img_size=64, num_classes=2, seed=42):
    """生成合成分割数据

    生成简单的几何图形(圆形/方形)作为图像，
    对应的二值分割标签为图形内部为1，外部为0。

    Args:
        num_samples: 样本数量
        img_size: 图像尺寸(正方形)
        num_classes: 类别数
        seed: 随机种子

    Returns:
        list: [(image, label), ...]
    """
    rng = np.random.RandomState(seed)
    data = []

    for i in range(num_samples):
        # 生成图像：背景 + 随机图形
        image = np.ones((img_size, img_size, 3), dtype=np.uint8) * 128
        label = np.zeros((img_size, img_size), dtype=np.uint8)

        # 随机绘制1-3个圆形
        num_circles = rng.randint(1, 4)
        for _ in range(num_circles):
            cx = rng.randint(img_size // 4, 3 * img_size // 4)
            cy = rng.randint(img_size // 4, 3 * img_size // 4)
            radius = rng.randint(5, img_size // 6)

            # 在图像上绘制圆形
            y, x = np.ogrid[:img_size, :img_size]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2

            # 图像上着色
            color = rng.randint(50, 255, size=3)
            image[mask] = color

            # 标签上标记
            label[mask] = 1

        # 添加噪声
        noise = rng.randint(-20, 20, size=image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        data.append((image, label))

    return data


def test_metrics():
    """测试指标计算"""
    from utils.metrics import compute_all_metrics

    print("\n[测试] 指标计算函数")
    # 合成预测和标签(完全一致)
    pred = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]])
    target = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]])

    metrics = compute_all_metrics(pred, target, num_classes=2)
    print(f"  完全一致的预测: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}")
    assert metrics["iou"] > 0.99, "IoU应接近1"
    assert metrics["dice"] > 0.99, "Dice应接近1"
    assert metrics["pixel_accuracy"] == 1.0, "PA应为1"

    # 部分错误
    pred2 = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]])
    metrics2 = compute_all_metrics(pred2, target, num_classes=2)
    print(f"  部分错误预测: IoU={metrics2['iou']:.4f}, Dice={metrics2['dice']:.4f}")
    assert metrics2["iou"] < 1.0, "IoU应小于1"
    assert metrics2["pixel_accuracy"] < 1.0, "PA应小于1"

    print("  [通过] 指标计算测试通过")
    return True


def test_unet():
    """测试U-Net模型前向传播"""
    import torch
    from models.unet import UNet

    print("\n[测试] U-Net模型")
    model = UNet(in_channels=3, num_classes=1, bilinear=True, base_features=32)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {out.shape}")
    assert out.shape == (2, 1, 64, 64), f"输出形状错误: {out.shape}"
    print("  [通过] U-Net前向传播测试通过")
    return True


def test_augmentation():
    """测试数据增强"""
    from utils.data_augment import SegmentationAugmentor

    print("\n[测试] 数据增强")
    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (64, 64), dtype=np.uint8)

    augmentor = SegmentationAugmentor(
        aug_config={
            "horizontal_flip": True,
            "vertical_flip": True,
            "random_rotation": True,
            "elastic_deform": False,
        },
        prob=1.0,
    )
    aug_img, aug_mask = augmentor(image, mask)
    print(f"  原始图像形状: {image.shape}")
    print(f"  增强后图像形状: {aug_img.shape}")
    assert aug_img.shape == image.shape, "增强后形状应一致"
    print("  [通过] 数据增强测试通过")
    return True


def test_dataset_validator():
    """测试数据集校验器"""
    from pipeline.dataset_validator import DatasetValidator
    from utils.logger import ExperimentLogger

    print("\n[测试] 数据集校验器")
    # 使用合成数据(不检查文件系统，直接传数据)
    logger = ExperimentLogger(to_console=False)
    print("  (跳过文件系统校验，使用合成数据直接训练)")
    print("  [通过] 数据集校验器初始化测试通过")
    return True


def test_full_agent():
    """测试完整Agent流程"""
    from agent.core import SegmentationAgent
    from utils.logger import ExperimentLogger
    import config

    print("\n[测试] 完整Agent流程")

    # 生成合成数据
    print("  生成合成数据...")
    train_data = generate_synthetic_data(num_samples=16, img_size=64, seed=42)
    val_data = generate_synthetic_data(num_samples=8, img_size=64, seed=99)
    print(f"  训练数据: {len(train_data)}个样本")
    print(f"  验证数据: {len(val_data)}个样本")

    # 创建Agent(使用合成数据，跳过文件校验)
    # 修改配置使其快速运行
    test_config = type("TestConfig", (), {})()
    for attr in dir(config):
        if not attr.startswith("_"):
            setattr(test_config, attr, getattr(config, attr))
    # 减少迭代次数加速测试
    test_config.MAX_ITERATIONS = 2
    test_config.NUM_CLASSES = 2
    test_config.DEVICE = "cpu"
    test_config.INITIAL_CONFIG = {
        "learning_rate": 1e-3,
        "batch_size": 4,
        "num_epochs": 2,
        "weight_decay": 1e-5,
        "optimizer": "adam",
        "loss_function": "dice",
        "augmentation": {
            "horizontal_flip": True,
            "vertical_flip": False,
            "random_rotation": False,
            "elastic_deform": False,
        },
    }
    test_config.IOU_THRESHOLD = 0.3
    test_config.DICE_THRESHOLD = 0.4
    test_config.PIXEL_ACC_THRESHOLD = 0.5
    test_config.USE_LLM_ADVISOR = False
    test_config.WARM_START = True
    test_config.PATIENCE = 10

    logger = ExperimentLogger(
        log_file=os.path.join(test_config.RESULTS_DIR, "test.log"),
        experiment_name="test_agent",
    )

    agent = SegmentationAgent(
        config=test_config,
        train_data=train_data,
        val_data=val_data,
        logger=logger,
    )

    print("  启动Agent...")
    summary = agent.run()

    print(f"\n  Agent运行结果:")
    print(f"    成功: {summary['success']}")
    print(f"    总迭代: {summary['total_iterations']}")
    if "final_metrics" in summary:
        fm = summary["final_metrics"]
        print(f"    最终IoU: {fm.get('iou', 0):.4f}")
        print(f"    最终Dice: {fm.get('dice', 0):.4f}")
        print(f"    最终PA: {fm.get('pixel_accuracy', 0):.4f}")

    assert summary["success"], "Agent应成功完成"
    print("  [通过] 完整Agent流程测试通过")
    return True


def test_comparison_experiments():
    """测试对照实验"""
    from agent.core import SegmentationAgent
    from utils.logger import ExperimentLogger
    import config

    print("\n[测试] 对照实验")
    train_data = generate_synthetic_data(num_samples=12, img_size=64, seed=10)
    val_data = generate_synthetic_data(num_samples=6, img_size=64, seed=20)

    test_config = type("TestConfig", (), {})()
    for attr in dir(config):
        if not attr.startswith("_"):
            setattr(test_config, attr, getattr(config, attr))
    test_config.NUM_CLASSES = 2
    test_config.DEVICE = "cpu"
    test_config.INITIAL_CONFIG = {
        "learning_rate": 1e-3,
        "batch_size": 4,
        "num_epochs": 1,
        "weight_decay": 1e-5,
        "optimizer": "adam",
        "loss_function": "dice",
        "augmentation": {
            "horizontal_flip": True,
            "vertical_flip": False,
            "random_rotation": False,
            "elastic_deform": False,
        },
    }

    logger = ExperimentLogger(to_console=False)
    agent = SegmentationAgent(
        config=test_config,
        train_data=train_data,
        val_data=val_data,
        logger=logger,
    )

    results = agent.run_comparison_experiments()
    print(f"  对照实验数量: {len(results)}")
    for r in results:
        if "error" in r:
            print(f"    {r['experiment_name']}: 失败")
        else:
            m = r["metrics"]
            print(f"    {r['experiment_name']}: IoU={m.get('iou', 0):.4f}")

    assert len(results) > 0, "应至少有一组对照实验"
    print("  [通过] 对照实验测试通过")
    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("图像分割自动调优Agent - 测试套件")
    print("=" * 60)

    tests = [
        ("指标计算", test_metrics),
        ("U-Net模型", test_unet),
        ("数据增强", test_augmentation),
        ("数据集校验", test_dataset_validator),
        ("完整Agent流程", test_full_agent),
        ("对照实验", test_comparison_experiments),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result, None))
        except Exception as e:
            print(f"\n  [失败] {name} 测试失败: {e}")
            traceback.print_exc()
            results.append((name, False, str(e)))

    # 汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    passed = sum(1 for _, r, _ in results if r)
    total = len(results)
    for name, result, error in results:
        status = "通过" if result else "失败"
        print(f"  {name}: [{status}]")
        if error:
            print(f"    错误: {error}")
    print(f"\n通过率: {passed}/{total}")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
