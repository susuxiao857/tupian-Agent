# -*- coding: utf-8 -*-
"""随机种子与可复现性。"""

import os
import random

import numpy as np


def set_seed(seed=42):
    """固定 Python / NumPy / PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
