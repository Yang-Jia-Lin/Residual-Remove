"""Configs/path.py"""

import platform
from pathlib import Path

if platform.system() == "Windows":
    BASE_DIR = Path("D:/Coding/Python/ResidualRemove")
    DATA_DIR = Path("D:/Coding/Python/0-Data")
else:
    BASE_DIR = Path("/root/autodl-tmp/ResidualRemove")
    DATA_DIR = Path("/root/autodl-tmp/0-Data")

RESULT_DIR = BASE_DIR / "Results"


if __name__ == "__main__":
    from Configs.path import DATA_DIR, RESULT_DIR

    print(DATA_DIR)               # /root/autodl-tmp/0-Data
    cifar_path = DATA_DIR / "cifar10"
    save_path  = RESULT_DIR / "resnet18" / "scalar"
    print(cifar_path)
    print(save_path)
    