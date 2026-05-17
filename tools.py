import warnings

warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated.*",
    category=FutureWarning,
)

import os
import platform
import random

import numpy as np
import torch


def set_device(device_num, use_cuda=False):
    system = platform.system()
    if not use_cuda:
        return "cpu"

    if system in ["Linux", "Windows"] and torch.cuda.is_available():
        return "cuda:" + str(device_num)
    if system == "Darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_dirs(paths: list) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)
