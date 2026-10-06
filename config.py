# -*- coding: utf-8 -*-
"""
配置文件
定义数据集路径、超参数初始值、评估阈值、迭代控制参数等全局配置。
"""

import os

# ==================== 路径配置 ====================
# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 数据集目录
DATA_DIR = os.path.join(BASE_DIR, "data")
# 图像子目录
IMAGE_DIR = os.path.join(DATA_DIR, "images")
# 标签子目录
LABEL_DIR = os.path.join(DATA_DIR, "labels")
# 结果输出目录
RESULTS_DIR = os.path.join(BASE_DIR, "results")
# 日志文件路径
LOG_FILE = os.path.join(RESULTS_DIR, "experiment.log")

# ==================== 数据集配置 ====================
# 支持的图像格式
SUPPORTED_IMAGE_FORMATS = (".png", ".jpg", ".jpeg", ".bmp", ".tif")
# 支持的标签格式
SUPPORTED_LABEL_FORMATS = (".png", ".tif")
# 图像尺寸(校验时要求一致，None表示不强制)
EXPECTED_IMAGE_SIZE = None  # 例如 (256, 256)
# 分类数: 1表示二分类，>1表示多分类
NUM_CLASSES = 2
# 二分类前景像素值范围(多分类时为[0, NUM_CLASSES-1])
LABEL_VALUE_RANGE = (0, 1)

# ==================== 模型配置 ====================
# U-Net初始特征数
INITIAL_FEATURES = 64
# 是否使用双卷积块
USE_BILINEAR_UPSAMPLE = True

# ==================== 训练超参数(初始值) ====================
INITIAL_CONFIG = {
    "learning_rate": 1e-3,          # 初始学习率
    "batch_size": 4,                # 批大小
    "num_epochs": 5,                # 每轮迭代训练轮数
    "weight_decay": 1e-5,          # 权重衰减
    "optimizer": "adam",            # 优化器类型
    "loss_function": "dice",       # 损失函数: dice / ce / dice_ce
    # 数据增强开关
    "augmentation": {
        "horizontal_flip": True,    # 水平翻转
        "vertical_flip": True,      # 垂直翻转
        "random_rotation": False,   # 随机旋转
        "elastic_deform": False,    # 弹性变形
    },
}

# ==================== 评估阈值 ====================
# 达到以下阈值则认为模型达标，停止迭代
IOU_THRESHOLD = 0.70       # IoU阈值
DICE_THRESHOLD = 0.75      # Dice系数阈值
PIXEL_ACC_THRESHOLD = 0.90 # 像素准确率阈值

# ==================== 迭代控制 ====================
MAX_ITERATIONS = 5          # 最大迭代轮数
# 学习率衰减因子(每轮迭代将学习率乘以此值)
LR_DECAY_FACTOR = 0.5
# 学习率下限(低于此值则停止衰减)
LR_MIN = 1e-6
# 指标平台期：连续 patience 轮 IoU 波动小于 MIN_DELTA 则提前停止
PATIENCE = 2
MIN_DELTA = 0.005
# 跨迭代热启动：后续轮次从上一轮最佳权重继续训练(而非每次随机初始化)
WARM_START = True
# 梯度裁剪阈值(0 表示不裁剪)
GRAD_CLIP_NORM = 1.0

# ==================== 设备配置 ====================
def _detect_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


DEVICE = _detect_device()

# ==================== 随机种子 ====================
RANDOM_SEED = 42

# ==================== 大模型(LLM)配置 ====================
# 是否启用大模型作为实验策略顾问(诊断训练状态、建议超参调整、生成分析报告)
# True: 接入 DeepSeek 等大模型做智能决策
# False: 使用规则驱动的调参策略(默认，无需 API Key)
USE_LLM_ADVISOR = True

# 当前使用的服务商(在 LLM_PROVIDERS 中选择键名)
# 可选: "deepseek" / "zhipu" / "qwen" / "openai" / "local"
CURRENT_PROVIDER = os.environ.get("LLM_PROVIDER", "deepseek")

# 各服务商配置(均为 OpenAI 兼容接口)
# 获取 DeepSeek API Key: https://platform.deepseek.com -> API Keys
LLM_PROVIDERS = {
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-flash",   # DeepSeek-V4.1-Flash(最新)
    },
    "zhipu": {
        # 智谱 GLM: https://open.bigmodel.cn
        "api_key": os.environ.get("ZHIPU_API_KEY", ""),
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-5.2",
    },
    "qwen": {
        # 通义千问: https://dashscope.console.aliyun.com
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3-max",
    },
    "openai": {
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5.6",
    },
    "local": {
        # 本地模拟模式：无 API Key 时使用规则模拟大模型决策
        "api_key": "",
        "base_url": "",
        "model": "local-simulator",
    },
}

_provider = LLM_PROVIDERS.get(CURRENT_PROVIDER, LLM_PROVIDERS["local"])
LLM_CONFIG = {
    "provider": CURRENT_PROVIDER,
    "api_key": _provider.get("api_key", ""),
    "base_url": _provider.get("base_url", ""),
    "model": os.environ.get("LLM_MODEL", _provider.get("model", "local-simulator")),
    "temperature": 0.3,        # 采样温度，实验决策建议偏低以保持确定性
    "max_tokens": 2048,        # 单次最大生成 token 数
    "request_timeout": 60,     # 请求超时(秒)
    # 思考模式开关(enabled/disabled，仅 DeepSeek 生效)；
    # 决策场景需解析 JSON 输出，关闭思考避免思维链占用 max_tokens
    "thinking": "disabled",
    "stream": False,           # 是否启用流式响应(决策场景建议关闭以便解析JSON)
    "max_iterations": 10,     # Agent 循环最大步数(防止无限迭代)
}

# 大模型决策的可信度阈值：大模型建议的配置若超出合理范围会被规则修正
LLM_SAFETY_BOUNDS = {
    "learning_rate": (1e-7, 1e-1),    # 学习率安全区间
    "batch_size": (1, 64),             # 批大小安全区间
    "num_epochs": (1, 50),              # 训练轮数安全区间
}
