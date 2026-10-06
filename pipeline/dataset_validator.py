# -*- coding: utf-8 -*-
"""
数据集校验模块
==================
检查图像/标签的格式、数量、尺寸、类别、数据完整性。
在Agent启动训练前自动执行，确保数据集满足训练要求。
"""

import os

import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from utils.logger import ExperimentLogger


class DatasetValidator:
    """数据集校验器

    校验项:
        1. 图像与标签数量是否匹配
        2. 文件格式是否支持(png/jpg/bmp/tif)
        3. 图像与标签尺寸是否一致
        4. 标签值是否在合法范围内
        5. 图像是否可正常读取(完整性检查)

    Args:
        image_dir: 图像目录
        label_dir: 标签目录
        logger: 日志记录器
        supported_image_formats: 支持的图像格式
        expected_size: 期望的图像尺寸(可选)
        label_value_range: 标签值合法范围
        num_classes: 类别数
    """

    def __init__(self, image_dir, label_dir, logger=None,
                 supported_image_formats=None, expected_size=None,
                 label_value_range=None, num_classes=2):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.logger = logger or ExperimentLogger()
        self.supported_image_formats = supported_image_formats or (".png", ".jpg", ".jpeg")
        self.expected_size = expected_size
        self.label_value_range = label_value_range or (0, num_classes - 1)
        self.num_classes = num_classes

    def validate(self):
        """执行完整数据集校验

        Returns:
            dict: 校验结果
                - valid: 是否通过校验
                - num_images: 图像数量
                - num_labels: 标签数量
                - issues: 问题列表
                - file_pairs: 有效的(图像, 标签)文件对列表
        """
        self.logger.info(f"开始数据集校验: 图像目录={self.image_dir}, 标签目录={self.label_dir}")

        issues = []

        # 1. 检查目录是否存在
        if not os.path.isdir(self.image_dir):
            issues.append(f"图像目录不存在: {self.image_dir}")
            self.logger.log_dataset_anomaly(f"图像目录不存在", self.image_dir)
            return {"valid": False, "issues": issues, "num_images": 0,
                    "num_labels": 0, "file_pairs": []}

        if not os.path.isdir(self.label_dir):
            issues.append(f"标签目录不存在: {self.label_dir}")
            self.logger.log_dataset_anomaly(f"标签目录不存在", self.label_dir)
            return {"valid": False, "issues": issues, "num_images": 0,
                    "num_labels": 0, "file_pairs": []}

        # 2. 获取文件列表
        image_files = self._get_files(self.image_dir, self.supported_image_formats)
        label_files = self._get_files(self.label_dir, self.supported_image_formats)

        self.logger.info(f"找到图像文件 {len(image_files)} 个, 标签文件 {len(label_files)} 个")

        # 3. 检查数量匹配
        if len(image_files) == 0:
            issues.append("图像目录中没有支持的图像文件")
            self.logger.log_dataset_anomaly("无图像文件", self.image_dir)
        if len(label_files) == 0:
            issues.append("标签目录中没有支持的标签文件")
            self.logger.log_dataset_anomaly("无标签文件", self.label_dir)
        if len(image_files) != len(label_files) and len(image_files) > 0 and len(label_files) > 0:
            issues.append(
                f"图像数量({len(image_files)})与标签数量({len(label_files)})不匹配"
            )
            self.logger.log_dataset_anomaly(
                f"数量不匹配: 图像{len(image_files)} vs 标签{len(label_files)}"
            )

        # 4. 匹配文件对并逐项校验
        file_pairs = []
        for img_path in image_files:
            label_path = self._match_label(img_path, label_files)
            if label_path is None:
                issues.append(f"未找到对应的标签文件: {os.path.basename(img_path)}")
                self.logger.log_dataset_anomaly("未匹配到标签", img_path)
                continue

            # 校验单对文件
            pair_issues = self._validate_pair(img_path, label_path)
            if pair_issues:
                issues.extend(pair_issues)
            else:
                file_pairs.append((img_path, label_path))

        is_valid = len(issues) == 0 and len(file_pairs) > 0
        self.logger.info(
            f"数据集校验完成: 通过={is_valid}, "
            f"有效文件对={len(file_pairs)}, 问题数={len(issues)}"
        )

        return {
            "valid": is_valid,
            "num_images": len(image_files),
            "num_labels": len(label_files),
            "issues": issues,
            "file_pairs": file_pairs,
        }

    def _get_files(self, directory, formats):
        """获取目录下指定格式的文件列表"""
        if not os.path.isdir(directory):
            return []
        files = []
        for f in os.listdir(directory):
            if f.lower().endswith(formats):
                files.append(os.path.join(directory, f))
        files.sort()
        return files

    def _match_label(self, image_path, label_files):
        """根据图像文件名匹配标签文件(去掉扩展名匹配)"""
        img_stem = os.path.splitext(os.path.basename(image_path))[0]
        for label_path in label_files:
            label_stem = os.path.splitext(os.path.basename(label_path))[0]
            if label_stem == img_stem:
                return label_path
        return None

    def _load_image(self, file_path):
        """加载图像为numpy数组"""
        if HAS_CV2:
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is not None and img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif HAS_PIL:
            img = np.array(Image.open(file_path))
            return img
        return None

    def _validate_pair(self, image_path, label_path):
        """校验一对图像-标签文件"""
        issues = []

        # 5. 完整性检查：能否正常读取
        try:
            image = self._load_image(image_path)
            if image is None:
                raise ValueError("图像读取结果为空")
        except Exception as e:
            msg = f"图像读取失败: {os.path.basename(image_path)} - {e}"
            issues.append(msg)
            self.logger.log_dataset_anomaly(str(e), image_path)
            return issues

        try:
            label = self._load_image(label_path)
            if label is None:
                raise ValueError("标签读取结果为空")
        except Exception as e:
            msg = f"标签读取失败: {os.path.basename(label_path)} - {e}"
            issues.append(msg)
            self.logger.log_dataset_anomaly(str(e), label_path)
            return issues

        # 6. 尺寸检查
        img_h, img_w = image.shape[:2]
        lbl_h, lbl_w = label.shape[:2]
        if img_h != lbl_h or img_w != lbl_w:
            issues.append(
                f"尺寸不匹配: {os.path.basename(image_path)} "
                f"图像({img_h}x{img_w}) vs 标签({lbl_h}x{lbl_w})"
            )
            self.logger.log_dataset_anomaly(
                f"尺寸不匹配 {img_h}x{img_w} vs {lbl_h}x{lbl_w}",
                image_path,
            )

        # 7. 期望尺寸检查
        if self.expected_size is not None:
            if (img_h, img_w) != self.expected_size:
                issues.append(
                    f"图像尺寸({img_h}x{img_w})与期望({self.expected_size[0]}x"
                    f"{self.expected_size[1]})不符"
                )

        # 8. 标签值范围检查
        label_vals = label.flatten()
        label_min, label_max = int(label_vals.min()), int(label_vals.max())
        expected_min, expected_max = self.label_value_range
        if label_min < expected_min or label_max > expected_max:
            issues.append(
                f"标签值范围[{label_min}, {label_max}]超出合法范围"
                f"[{expected_min}, {expected_max}]: {os.path.basename(label_path)}"
            )
            self.logger.log_dataset_anomaly(
                f"标签值范围[{label_min},{label_max}]不合法(期望[{expected_min},{expected_max}])",
                label_path,
            )

        # 9. 标签通道检查(应为单通道)
        if label.ndim == 3 and label.shape[2] != 1:
            self.logger.debug(
                f"标签为多通道，已转为单通道: {os.path.basename(label_path)}"
            )

        return issues

    def load_pairs(self, file_pairs):
        """将校验通过的文件对加载为内存数据集 [(image, label), ...]。"""
        data = []
        for img_path, label_path in file_pairs:
            image = self._load_image(img_path)
            label = self._load_image(label_path)
            if image is None or label is None:
                continue
            if label.ndim == 3:
                label = label[:, :, 0] if label.shape[2] == 1 else label.mean(axis=2)
            label = np.asarray(label, dtype=np.uint8)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            data.append((np.asarray(image), label))
        self.logger.info(f"已从磁盘加载 {len(data)} 对图像-标签")
        return data
