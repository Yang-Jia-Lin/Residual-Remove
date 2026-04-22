"""Configs/paras.py"""
import platform
from pathlib import Path


if platform.system() == "Windows":
    BASE_DIR = Path("D:/Coding/Python/ResidualRemove")
    DATA_DIR = Path("D:/Coding/Python/0-Data")
else:
    BASE_DIR = Path("/root/autodl-tmp/ResidualRemove")
    DATA_DIR = Path("/root/autodl-tmp/0-Data")

# RESULT_DIR = BASE_DIR / "Results"
RESULT_DIR_1 = BASE_DIR / "Results" / "Exp1_Motivation"
RESULT_DIR_2 = BASE_DIR / "Results" / "Exp2_Compensator"
# RESULT_DIR_3 = BASE_DIR / "Results" / "Exp3_Motivation"
# RESULT_DIR_4 = BASE_DIR / "Results" / "Exp4_Motivation"

COLORS = {
    'grey':   '#999999',
    'brown':  '#8D574B',
    'green':  '#2ca02c',
    'purple': '#9467bd',
    'red':    '#d62728',
    'blue':   '#1f77b4',
}