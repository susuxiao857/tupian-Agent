# 面向图像分割任务的自主迭代优化 Agent

> 接入 DeepSeek 大模型作为实验策略顾问，将 U-Net 分割训练流程封装为可调度单元，由大模型自主诊断训练状态、决策超参调整、判断是否继续迭代，并生成实验分析报告。

## 项目背景

图像分割实验包含数据集准备、模型训练、指标评估、超参调整等多个环节，每轮迭代都需要人工介入。本项目开发面向图像分割实验的 Agent 智能体，将 U-Net 训练流程封装为可调度单元，由大模型（DeepSeek）作为"实验策略大脑"自主决策调参方案，并配套异常捕获与日志记录，实现分割实验全流程自动化。

**核心能力：**
- **大模型实验策略顾问**：DeepSeek 基于训练指标诊断训练状态（过拟合/欠拟合/发散/正常），自主建议超参调整方案
- **结构化 JSON 决策**：大模型输出 `{diagnosis, reasoning, new_config, should_continue, suggestions}` JSON，由 Agent 解析执行
- **安全边界修正**：大模型建议的超参若超出合理范围会被规则修正，防止发散
- **智能终止决策**：大模型综合指标趋势与剩余预算判断是否继续迭代
- **实验分析报告**：大模型用自然语言总结实验发现与结论
- **规则降级**：无 API Key 时降级为规则驱动的调参策略，保证最小可运行

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│              启动 SegmentationAgent                          │
└────────────────────────────┬────────────────────────────────┘
                             ▼
              ┌──────────────────────────┐
              │  数据集校验               │ ← DatasetValidator
              │  (检查格式/尺寸/标签范围)  │
              └────────────┬─────────────┘
                           ▼
              ┌──────────────────────────┐
              │  训练 U-Net               │ ← UNetTrainer (可调度单元)
              │  (前向/损失/反向/优化)     │
              └────────────┬─────────────┘
                           ▼
              ┌──────────────────────────┐
              │  评估指标                 │ ← Evaluator
              │  (IoU / Dice / PA)        │
              └────────────┬─────────────┘
                           ▼
         ┌─────────────────────────────────────┐
         │  LLMAdvisor (DeepSeek 大模型顾问)    │
         │                                      │
         │  输入: 指标 + 当前配置 + 历史趋势     │
         │  输出: JSON 决策                      │
         │   {                                   │
         │     diagnosis: 训练状态诊断           │
         │     reasoning: 决策依据               │
         │     new_config: 新超参方案            │
         │     should_continue: 是否继续         │
         │     suggestions: 下一步建议           │
         │   }                                   │
         │  + 安全边界修正(防止发散)             │
         └────────────┬────────────────────────┘
                      ▼
              ┌──────────────────────────┐
              │  终止判断                 │ ← TaskPlanner
              │  (规则达标 OR 大模型建议)  │
              └────────────┬─────────────┘
                 ╱          ╲
           达标/最大       未达标
               │              │
               ▼              └──→ 回到训练步骤
        ┌────────────────────┐
        │  大模型生成分析报告  │ ← LLMAdvisor.generate_report
        │  + 实验指标汇总输出  │
        └────────────────────┘
```

## 项目结构

```
项目二_面向图像分割任务的自主迭代优化Agent/
├── agent/                    # Agent 智能体模块
│   ├── __init__.py
│   ├── core.py              # SegmentationAgent: 任务循环主控(含大模型接入)
│   └── task_planner.py      # 任务规划: 迭代策略、终止条件、对照实验
├── pipeline/                # 流程模块(可调度单元)
│   ├── __init__.py
│   ├── dataset_validator.py # 数据集校验
│   ├── trainer.py           # U-Net 训练封装
│   ├── evaluator.py         # 指标评估
│   └── hyperparam_tuner.py  # 超参调优(规则 + 大模型双模式)
├── models/                  # 模型定义
│   ├── __init__.py
│   └── unet.py             # U-Net 网络
├── llm/                     # 大模型模块
│   ├── __init__.py
│   └── llm_advisor.py      # LLMAdvisor: 大模型实验策略顾问
├── utils/                   # 工具模块
│   ├── __init__.py
│   ├── metrics.py          # IoU/Dice/像素准确率
│   ├── logger.py           # 实验日志记录器
│   └── data_augment.py     # 数据增强策略
├── data/                    # 数据集目录
│   └── README.md           # 数据格式说明
├── tests/                   # 测试
│   ├── __init__.py
│   └── test_agent.py       # 合成数据测试套件
├── results/                 # 实验结果输出
│   └── .gitkeep
├── config.py                # 全局配置(含 DeepSeek 大模型配置)
├── main.py                  # 入口脚本
├── requirements.txt         # 依赖清单
└── README.md
```

## 核心模块说明

### 1. SegmentationAgent（`agent/core.py`）

Agent 任务循环主控，编排完整流程并接入大模型：

```python
from agent.core import SegmentationAgent
import config

agent = SegmentationAgent(config=config, train_data=..., val_data=...)
summary = agent.run()                  # 自动调优主流程(含大模型决策)
results = agent.run_comparison_experiments()  # 对照实验
```

**大模型接入点：**
- **调参决策**：每轮迭代后调用 `LLMAdvisor.advise_tuning()` 获取大模型决策的超参方案
- **终止决策**：将大模型的 `should_continue` 建议注入 `TaskPlanner`，综合规则阈值双重判断
- **分析报告**：实验结束后调用 `LLMAdvisor.generate_report()` 生成自然语言分析报告

### 2. LLMAdvisor（`llm/llm_advisor.py`）

大模型实验策略顾问，封装 DeepSeek API 调用：

| 方法 | 说明 |
|------|------|
| `advise_tuning(metrics, current_config, history, thresholds, iteration, max_iter)` | 建议下一轮超参调整方案，返回 JSON 决策 |
| `generate_report(summary, history)` | 生成实验分析报告（自然语言） |
| `_parse_json_response(text)` | 解析大模型 JSON 输出（容错 markdown 代码块） |
| `_apply_safety_bounds(new_config)` | 对大模型建议的配置做安全边界修正 |

**大模型输出 JSON 结构：**
```json
{
  "diagnosis": "接近达标",
  "reasoning": "IoU=0.68 接近阈值 0.70，需精调学习率",
  "new_config": {
    "learning_rate": 0.0007,
    "batch_size": 4,
    "num_epochs": 7,
    "augmentation": {"horizontal_flip": true, ...}
  },
  "should_continue": true,
  "suggestions": "轻微降低学习率，增加训练轮数"
}
```

### 3. 超参调优（`pipeline/hyperparam_tuner.py`）

双模式调优器：
- **大模型模式**（注入 `llm_advisor`）：由 DeepSeek 自主决策超参调整方案
- **规则模式**（无 `llm_advisor`）：基于 if-else 规则衰减学习率、调整数据增强
- 大模型决策失败时自动降级规则模式

### 4. 任务规划与终止判断（`agent/task_planner.py`）

终止逻辑（优先级从高到低）：
1. 指标达标（IoU + Dice 同时达标）→ 终止
2. 达到最大迭代次数 → 终止
3. 大模型建议终止（且未达标）→ 终止
4. 否则继续

### 5. U-Net 模型（`models/unet.py`）

标准 U-Net 实现：编码器 4 次下采样 + 瓶颈层 + 解码器 4 次上采样 + 跳跃连接，支持二分类和多分类。

### 6. 评估指标（`utils/metrics.py`）

| 指标 | 公式 | 说明 |
|------|------|------|
| IoU (Jaccard) | `|P∩G| / |P∪G|` | 衡量重叠度 |
| Dice (F1) | `2|P∩G| / (|P|+|G|)` | 对边界敏感 |
| Pixel Accuracy | 正确像素数 / 总像素数 | 整体准确率 |

### 7. 安全边界（`config.py`）

大模型建议的超参若超出合理范围会被修正：

| 参数 | 安全区间 |
|------|---------|
| learning_rate | (1e-7, 1e-1) |
| batch_size | (1, 64) |
| num_epochs | (1, 50) |

## 快速开始

### 1. 安装依赖

```bash
cd 项目二_面向图像分割任务的自主迭代优化Agent
pip install -r requirements.txt
```

### 2. 配置 DeepSeek API Key（可选，启用智能模式）

**方式一：环境变量（推荐）**

PowerShell：
```powershell
$env:DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxx"
python main.py
```

**方式二：编辑 config.py**

设置 `USE_LLM_ADVISOR = True`（默认已启用），API Key 通过环境变量或 config.py 填入。

**获取 API Key：** https://platform.deepseek.com → API Keys

### 3. 运行自动调优（合成数据）

无需真实数据集，使用合成数据即可体验完整 Agent 流程：

```bash
python main.py
```

无 API Key 时自动降级为本地模拟模式（规则驱动调参），仍可完整运行。

### 4. 运行测试

```bash
python tests/test_agent.py
```

测试覆盖：指标计算、U-Net 模型、数据增强、数据集校验、完整 Agent 流程、对照实验。

### 5. 命令行参数

```bash
python main.py --mode both          # 自动调优 + 对照实验
python main.py --mode compare        # 仅对照实验
python main.py --num_samples 40      # 指定合成数据样本数
python main.py --img_size 128        # 指定合成图像尺寸
```

### 6. 切换其他大模型

编辑 `config.py` 中 `CURRENT_PROVIDER`：
- `"deepseek"` — DeepSeek-V4.1-Flash（最新，默认）
- `"zhipu"` — 智谱 GLM-5.2
- `"qwen"` — 通义千问 Qwen3-Max
- `"openai"` — GPT-5.6
- `"local"` — 本地模拟模式（无需 API Key）

## 配置说明（`config.py`）

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `USE_LLM_ADVISOR` | 是否启用大模型顾问 | `True` |
| `CURRENT_PROVIDER` | 大模型服务商 | `"deepseek"` |
| `LLM_PROVIDERS` | 各服务商配置（API Key、base_url、model） | — |
| `LLM_CONFIG["temperature"]` | 采样温度（决策建议偏低） | `0.3` |
| `LLM_SAFETY_BOUNDS` | 大模型建议的安全边界 | — |
| `IMAGE_DIR` / `LABEL_DIR` | 数据集目录 | `data/images` / `data/labels` |
| `NUM_CLASSES` | 类别数 | 2 |
| `INITIAL_CONFIG` | 初始超参 | lr=1e-3, bs=4, epochs=5 |
| `IOU_THRESHOLD` | IoU 达标阈值 | 0.70 |
| `DICE_THRESHOLD` | Dice 达标阈值 | 0.75 |
| `MAX_ITERATIONS` | 最大迭代次数 | 5 |
| `LR_DECAY_FACTOR` | 学习率衰减因子（规则模式） | 0.5 |
| `DEVICE` | 计算设备 | cuda（无 GPU 自动回退 cpu） |

## 输出结果

运行后结果保存在 `results/` 目录：

| 文件 | 说明 |
|------|------|
| `experiment.log` | 完整实验日志（训练过程、大模型决策、指标、异常） |
| `summary_report.txt` | 实验汇总报告 + **大模型分析报告** |
| `test.log` | 测试日志 |

汇总报告末尾会追加**大模型生成的实验分析报告**，包含：
1. 实验总体评价
2. 指标趋势分析
3. 超参调整效果
4. 关键发现与建议

## 降级策略

| 组件 | 降级链路 |
|------|---------|
| 大模型顾问 | DeepSeek API → 无 API Key 时降级本地模拟模式（规则驱动） |
| 大模型决策失败 | API 异常 → 自动降级规则调参模式 |
| 大模型建议越界 | 安全边界修正 → 学习率/批大小/训练轮数钳制到合理区间 |

## 技术栈

| 技术 | 用途 |
|------|------|
| **DeepSeek 大模型** | 实验策略顾问（诊断/调参/终止决策/分析报告） |
| Python | 编程语言 |
| PyTorch | 深度学习框架 |
| U-Net | 语义分割网络 |
| OpenCV | 图像处理 |
| NumPy | 数值计算 |
| Matplotlib | 可视化 |
| tqdm | 进度条 |

## 项目成果

- 搭建由大模型驱动的图像分割自动调优 Agent 原型
- 实现大模型作为实验策略大脑：诊断训练状态 + 自主调参 + 终止决策 + 分析报告
- 熟悉实验类 Agent 任务循环设计、异常处理
- 掌握智能体和计算机视觉任务结合的落地实现
- 形成数据集校验 → 训练 → 评估 → 大模型决策 → 迭代 → 汇总的完整闭环
