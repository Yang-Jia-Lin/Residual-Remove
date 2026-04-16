"""
Src/paras.py
"""
import platform
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

if platform.system() == "Windows":
    BASE_DRIVE = Path("D:/Coding/Python/DSCI")
else:
    BASE_DRIVE = Path("/workspace/user/Coding/jialin/DSCI")


COLORS = {
    'grey': '#999999',
    'brown': '#8D574B',
    'green': '#2ca02c',
    'purple': '#9467bd',
    'red': '#d62728',
    'blue': '#1f77b4'
}

