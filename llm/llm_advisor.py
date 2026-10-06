# -*- coding: utf-8 -*-
"""大模型实验策略顾问(LLMAdvisor)
================================
将大模型作为图像分割实验的"智能大脑"，提供以下能力：

1. 诊断训练状态：基于指标与 loss 曲线判断过拟合/欠拟合/发散/正常
2. 建议超参调整：自主决策学习率、数据增强、训练轮数等调整方案
3. 决策是否继续迭代：综合指标趋势与剩余预算判断
4. 生成实验分析报告：用自然语言总结实验发现与结论
5. 建议对照实验方案：规划有价值的对照实验组合

大模型通过结构化 JSON 输出决策，由 Agent 解析执行。
无 API Key 时降级为规则模拟模式，保证最小可运行。

接入方式：基于 LangChain (langchain-openai) 调用 OpenAI 兼容接口
(默认 DeepSeek，可切换智谱/通义/OpenAI)。
"""
import copy
import json
import re

from config import LLM_CONFIG, LLM_SAFETY_BOUNDS
from utils.logger import ExperimentLogger


# ==================== LangChain 依赖(惰性导入，本地模式不强制) ====================
# 仅在真正调用大模型时才导入，确保无 langchain 环境下本地模拟模式仍可运行


# ==================== 提示词模板 ====================

# 系统提示：定义大模型的角色与输出规范
SYSTEM_PROMPT = """你是一个图像分割实验的 AI 策略顾问。
你的职责是：基于 U-Net 分割训练的指标数据，给出专业的实验决策。

输出要求：
- 严格输出 JSON 格式(不要有任何额外文字、不要 markdown 代码块标记)
- 字段必须完整，数值需在合理范围内
- 分析与建议用简洁专业的中文

JSON 字段说明：
{
  "diagnosis": "训练状态诊断(过拟合/欠拟合/发散/正常等)",
  "reasoning": "决策依据(指标分析、趋势判断)",
  "new_config": {
    "learning_rate": 新学习率(float),
    "batch_size": 批大小(int),
    "num_epochs": 训练轮数(int),
    "augmentation": {
      "horizontal_flip": bool,
      "vertical_flip": bool,
      "random_rotation": bool,
      "elastic_deform": bool
    }
  },
  "should_continue": 是否继续迭代(bool),
  "suggestions": "下一步建议(简短)"
}"""


def _build_tuning_prompt(metrics, current_config, history, thresholds, iteration, max_iter):
    """构建超参调优请求的用户提示词。"""
    history_text = ""
    if history:
        lines = []
        for h in history[-5:]:  # 最近5轮
            m = h.get("metrics", {})
            lines.append(
                f"  迭代{h.get('iteration','?')}: "
                f"IoU={m.get('iou',0):.4f} Dice={m.get('dice',0):.4f} "
                f"PA={m.get('pixel_accuracy',0):.4f} loss={h.get('loss',0):.4f}"
            )
        history_text = "\n".join(lines)
    else:
        history_text = "  (首轮迭代，无历史)"

    return f"""请基于以下分割训练数据，给出下一轮迭代的超参调整决策。

【当前指标(验证集)】
- IoU: {metrics.get('iou', 0):.4f} (阈值 {thresholds.get('iou', 0.7)})
- Dice: {metrics.get('dice', 0):.4f} (阈值 {thresholds.get('dice', 0.75)})
- Pixel Accuracy: {metrics.get('pixel_accuracy', 0):.4f} (阈值 {thresholds.get('pixel_accuracy', 0.9)})
- 训练 loss: {metrics.get('loss', 0):.4f}

【当前超参配置】
{json.dumps(current_config, ensure_ascii=False, indent=2)}

【历史指标】
{history_text}

【迭代进度】第 {iteration} 轮 / 最多 {max_iter} 轮

请输出 JSON 决策(包含 diagnosis, reasoning, new_config, should_continue, suggestions)。"""


def _build_report_prompt(summary, history):
    """构建实验分析报告生成的用户提示词。"""
    history_text = ""
    if history:
        lines = []
        for h in history:
            m = h.get("metrics", {})
            lines.append(
                f"  迭代{h.get('iteration','?')}: "
                f"IoU={m.get('iou',0):.4f} Dice={m.get('dice',0):.4f} "
                f"loss={h.get('loss',0):.4f}"
            )
        history_text = "\n".join(lines)

    final = summary.get("final_metrics", {})
    return f"""请基于以下实验数据，生成一份简洁的实验分析报告。

【实验状态】{'成功' if summary.get('success') else '失败'}
【终止原因】{summary.get('reason', 'N/A')}
【总迭代轮数】{summary.get('total_iterations', 0)}
【最终指标】IoU={final.get('iou',0):.4f} Dice={final.get('dice',0):.4f} PA={final.get('pixel_accuracy',0):.4f}
【最终超参】{json.dumps(summary.get('final_config',{}), ensure_ascii=False)}
【迭代历史】
{history_text}

请输出一份包含以下内容的中文分析报告(纯文本，不要JSON)：
1. 实验总体评价
2. 指标趋势分析
3. 超参调整效果
4. 关键发现与建议"""


class LLMAdvisor:
    """大模型实验策略顾问

    作为图像分割实验的智能决策大脑，替代纯规则驱动的调参逻辑。

    Args:
        config: LLM 配置字典(默认取 config.LLM_CONFIG)
        logger: 日志记录器
        safety_bounds: 大模型建议的安全边界(默认取 config.LLM_SAFETY_BOUNDS)
    """

    def __init__(self, config=None, logger=None, safety_bounds=None):
        self.config = config or LLM_CONFIG
        self.logger = logger or ExperimentLogger()
        self.safety_bounds = safety_bounds or LLM_SAFETY_BOUNDS
        self.provider = self.config.get("provider", "local")
        self.api_key = self.config.get("api_key", "")
        self.base_url = self.config.get("base_url", "")
        self.model = self.config.get("model", "local-simulator")
        # 无 API Key 时降级本地模拟
        self.is_local = (
            self.provider == "local"
            or not self.api_key
            or not self.base_url
        )
        self._llm = None  # LangChain ChatOpenAI 实例(惰性初始化)

    # ==================== 对外主接口 ====================

    def advise_tuning(self, metrics, current_config, history, thresholds,
                      iteration, max_iter):
        """建议下一轮超参调整方案

        Args:
            metrics: 当前评估指标字典
            current_config: 当前超参配置
            history: 历史迭代记录列表
            thresholds: 目标阈值字典
            iteration: 当前迭代轮次(1-based)
            max_iter: 最大迭代轮数

        Returns:
            dict: {
                diagnosis, reasoning, new_config, should_continue, suggestions,
                source: "llm" 或 "local"
            }
        """
        if self.is_local:
            return self._local_advise_tuning(
                metrics, current_config, thresholds, iteration, max_iter
            )

        # 构建提示词
        user_prompt = _build_tuning_prompt(
            metrics, current_config, history, thresholds, iteration, max_iter
        )

        # 调用大模型
        try:
            raw = self._call_llm(user_prompt)
            decision = self._parse_json_response(raw)
            merged = self._merge_config(
                current_config, decision.get("new_config")
            )
            decision["new_config"] = self._apply_safety_bounds(merged)
            decision["source"] = "llm"
            self.logger.info(f"[大模型顾问] 诊断: {decision.get('diagnosis','')}")
            self.logger.info(f"[大模型顾问] 建议: {decision.get('suggestions','')}")
            return decision
        except Exception as e:
            self.logger.warning(f"大模型决策失败，降级规则模式: {e}")
            fallback = self._local_advise_tuning(
                metrics, current_config, thresholds, iteration, max_iter
            )
            fallback["source"] = "local_fallback"
            fallback["error"] = str(e)
            return fallback

    def generate_report(self, summary, history):
        """生成实验分析报告

        Args:
            summary: 实验汇总字典
            history: 历史迭代记录

        Returns:
            str: 自然语言分析报告
        """
        if self.is_local:
            return self._local_generate_report(summary, history)

        user_prompt = _build_report_prompt(summary, history)
        try:
            report = self._call_llm(user_prompt, system=False)
            self.logger.info("[大模型顾问] 实验分析报告已生成")
            return report.strip()
        except Exception as e:
            self.logger.warning(f"大模型报告生成失败，降级规则模式: {e}")
            return self._local_generate_report(summary, history)

    # ==================== LangChain 调用底层 ====================

    def _get_llm(self):
        """惰性初始化 LangChain ChatOpenAI 模型

        使用 langchain-openai 的 ChatOpenAI 封装，统一对接 OpenAI 兼容接口。
        DeepSeek 思考模式通过 model_kwargs 透传 extra_body。
        """
        if self._llm is None:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                raise RuntimeError(
                    "未安装 langchain-openai，请执行: "
                    "pip install langchain-openai"
                )

            # DeepSeek 新版模型默认开启思考模式，决策场景需解析 JSON 输出，
            # 显式关闭思考避免思维链占用 max_tokens
            extra_body = None
            if (self.provider == "deepseek"
                    and str(self.model).startswith("deepseek")):
                extra_body = {
                    "thinking": {"type": self.config.get("thinking", "disabled")}
                }

            self._llm = ChatOpenAI(
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.config.get("temperature", 0.3),
                max_tokens=self.config.get("max_tokens", 2048),
                timeout=self.config.get("request_timeout", 60),
                extra_body=extra_body,
            )
        return self._llm

    def _build_tuning_chain(self):
        """构建超参调优链路：system(角色+JSON规范) + human(指标数据) -> LLM

        注意：SYSTEM_PROMPT 含 JSON 花括号，必须用 SystemMessage 直接传入
        （不做模板格式化），否则花括号会被误判为模板变量。
        """
        from langchain_core.messages import SystemMessage
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=SYSTEM_PROMPT),
            ("human", "{user_prompt}"),
        ])
        return prompt | self._get_llm()

    def _build_report_chain(self):
        """构建报告生成链路：仅 human(实验数据) -> LLM(无 system 约束)"""
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("human", "{user_prompt}"),
        ])
        return prompt | self._get_llm()

    def _call_llm(self, user_prompt, system=True):
        """调用大模型，返回文本响应

        使用 LangChain LCEL 链路执行，AIMessage.content 即为响应文本。

        Args:
            user_prompt: 用户提示词
            system: 是否注入系统提示词(调优场景注入，报告场景不注入)

        Returns:
            str: 大模型响应文本
        """
        chain = self._build_tuning_chain() if system else self._build_report_chain()
        response = chain.invoke({"user_prompt": user_prompt})
        return response.content

    @staticmethod
    def _parse_json_response(text):
        """解析大模型返回的 JSON(容错处理 markdown 代码块)"""
        # 去除可能的 markdown 代码块标记
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        # 尝试提取 JSON 主体
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0)
        return json.loads(text)

    @staticmethod
    def _merge_config(current_config, suggested):
        """将大模型建议合并进当前配置，避免只返回部分字段时丢失超参。"""
        merged = copy.deepcopy(current_config) if current_config else {}
        if not isinstance(suggested, dict):
            return merged
        for key, value in suggested.items():
            if key == "augmentation" and isinstance(value, dict):
                merged.setdefault("augmentation", {})
                if isinstance(merged["augmentation"], dict):
                    merged["augmentation"].update(value)
                else:
                    merged["augmentation"] = dict(value)
            else:
                merged[key] = value
        return merged

    def _apply_safety_bounds(self, new_config):
        """对大模型建议的配置做安全边界修正

        防止大模型输出不合理的数值(如学习率过大导致发散)。
        """
        safe_config = copy.deepcopy(new_config)
        bounds = self.safety_bounds

        # 学习率
        if "learning_rate" in safe_config:
            lo, hi = bounds.get("learning_rate", (1e-7, 1e-1))
            lr = float(safe_config["learning_rate"])
            safe_config["learning_rate"] = max(lo, min(hi, lr))

        # 批大小
        if "batch_size" in safe_config:
            lo, hi = bounds.get("batch_size", (1, 64))
            bs = int(safe_config["batch_size"])
            safe_config["batch_size"] = max(lo, min(hi, bs))

        # 训练轮数
        if "num_epochs" in safe_config:
            lo, hi = bounds.get("num_epochs", (1, 50))
            ep = int(safe_config["num_epochs"])
            safe_config["num_epochs"] = max(lo, min(hi, ep))

        # 数据增强规范化为 bool
        aug = safe_config.get("augmentation", {})
        for key in ["horizontal_flip", "vertical_flip",
                    "random_rotation", "elastic_deform"]:
            if key in aug:
                aug[key] = bool(aug[key])
        safe_config["augmentation"] = aug

        return safe_config

    # ==================== 本地模拟模式 ====================

    def _local_advise_tuning(self, metrics, current_config, thresholds,
                              iteration, max_iter):
        """本地模拟模式：基于规则生成与大模型等价的决策结构

        无 API Key 时使用，保证流程可运行。
        """
        iou = metrics.get("iou", 0.0)
        dice = metrics.get("dice", 0.0)
        pa = metrics.get("pixel_accuracy", 0.0)
        loss = metrics.get("loss", 0.0)
        iou_th = thresholds.get("iou", 0.7)
        dice_th = thresholds.get("dice", 0.75)

        new_config = copy.deepcopy(current_config)
        current_lr = current_config.get("learning_rate", 1e-3)

        # 诊断
        if iou >= iou_th and dice >= dice_th:
            diagnosis = "正常(已达标)"
            reasoning = f"IoU={iou:.4f}>={iou_th}, Dice={dice:.4f}>={dice_th}, 指标达标"
            should_continue = False
            suggestions = "指标已达标，可停止迭代"
            # 轻微衰减学习率精调
            new_config["learning_rate"] = max(current_lr * 0.7, 1e-7)
        elif iou < 0.3:
            diagnosis = "欠拟合/发散"
            reasoning = f"IoU={iou:.4f} 远低于阈值 {iou_th}, 可能学习率过大或欠拟合"
            should_continue = iteration < max_iter
            new_config["learning_rate"] = max(current_lr * 0.3, 1e-7)
            suggestions = "大幅降低学习率，关闭复杂数据增强，关注训练拟合"
            aug = new_config.get("augmentation", {})
            for k in ["random_rotation", "elastic_deform"]:
                aug[k] = False
            new_config["augmentation"] = aug
        elif iou < iou_th * 0.7:
            diagnosis = "欠拟合"
            reasoning = f"IoU={iou:.4f} 低于阈值 {iou_th} 的 70%, 模型拟合不足"
            should_continue = iteration < max_iter
            new_config["learning_rate"] = max(current_lr * 0.5, 1e-7)
            new_config["num_epochs"] = min(
                new_config.get("num_epochs", 5) + 2, 50
            )
            suggestions = "降低学习率，增加训练轮数，减弱数据增强以提升拟合"
            aug = new_config.get("augmentation", {})
            for k in ["random_rotation", "elastic_deform"]:
                aug[k] = False
            new_config["augmentation"] = aug
        elif iou < iou_th:
            diagnosis = "接近达标"
            reasoning = f"IoU={iou:.4f} 接近阈值 {iou_th}, 需精调"
            should_continue = iteration < max_iter
            new_config["learning_rate"] = max(current_lr * 0.7, 1e-7)
            new_config["num_epochs"] = min(
                new_config.get("num_epochs", 5) + 2, 50
            )
            suggestions = "轻微降低学习率，增加训练轮数，保持增强策略"
        else:
            diagnosis = "正常"
            reasoning = "指标趋势正常"
            should_continue = iteration < max_iter
            new_config["learning_rate"] = max(current_lr * 0.8, 1e-7)
            suggestions = "继续迭代观察趋势"

        return {
            "diagnosis": diagnosis,
            "reasoning": reasoning,
            "new_config": new_config,
            "should_continue": should_continue,
            "suggestions": suggestions,
            "source": "local",
        }

    def _local_generate_report(self, summary, history):
        """本地模拟模式：生成简化分析报告"""
        final = summary.get("final_metrics", {})
        lines = ["=" * 50, "实验分析报告(本地模拟)", "=" * 50]
        lines.append(f"实验状态: {'成功' if summary.get('success') else '失败'}")
        lines.append(f"终止原因: {summary.get('reason', 'N/A')}")
        lines.append(f"总迭代轮数: {summary.get('total_iterations', 0)}")
        lines.append("")
        lines.append("最终指标:")
        lines.append(f"  IoU: {final.get('iou', 0):.4f}")
        lines.append(f"  Dice: {final.get('dice', 0):.4f}")
        lines.append(f"  Pixel Accuracy: {final.get('pixel_accuracy', 0):.4f}")
        lines.append("")
        if history:
            lines.append("指标趋势:")
            for h in history:
                m = h.get("metrics", {})
                lines.append(
                    f"  迭代{h.get('iteration','?')}: "
                    f"IoU={m.get('iou',0):.4f} "
                    f"Dice={m.get('dice',0):.4f} "
                    f"loss={h.get('loss',0):.4f}"
                )
        lines.append("")
        lines.append(
            "(配置 DeepSeek API Key 后可由大模型生成更深入的分析报告)"
        )
        return "\n".join(lines)
