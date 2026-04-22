"""Configs/paras.py"""

from dataclasses import dataclass, field
from typing import Dict, List


# ── 颜色 ─────────────────────────────────────────────────────────────
COLORS = {
    'grey':   '#999999',
    'brown':  '#8D574B',
    'green':  '#2ca02c',
    'purple': '#9467bd',
    'red':    '#d62728',
    'blue':   '#1f77b4',
}

# ── 模型 ─────────────────────────────────────────────────────────────
MODEL_ZOO = {
    "resnet18":    {"family": "resnet",    "depth": 18,  "num_classes": 10,   "input_size": 32,  "block_variant": "basic"},
    "resnet50":    {"family": "resnet",    "depth": 50,  "num_classes": 1000, "input_size": 224, "block_variant": "bottleneck"},
    "mobilenet_v2":{"family": "mobilenet", "width_mult": 1.0, "num_classes": 10, "input_size": 32},
}

# ── 补偿器配置 ────────────────────────────────────────────────────────
COMPENSATOR_CONFIG = {
    "scalar":    {"enabled": True},
    "affine":    {"enabled": True},
    "linear1x1": {"enabled": True},
    "low_rank":  {"enabled": True, "rank": 16},
    "adapter":   {"enabled": True, "rank": 16, "activation": "gelu"},
}

# ── Dataclass 配置 ────────────────────────────────────────────────────
@dataclass
class HardwareConfig:
    seed:        int  = 42
    device:      str  = "cuda"
    num_workers: int  = 4
    pin_memory:  bool = True

@dataclass
class DatasetConfig:
    default_dataset:      str  = "imagenet100"
    default_image_size:   int  = 32
    synthetic_if_missing: bool = True
    synthetic_train_size: int  = 512
    synthetic_val_size:   int  = 256
    calib_num_samples:    int  = 1024
    calib_batch_size:     int  = 64

@dataclass
class TrainingConfig:
    lr:                  float = 1e-3
    weight_decay:        float = 0.0
    epochs:              int   = 3
    batch_size:          int   = 32
    feature_loss_weight: float = 1.0
    logit_loss_weight:   float = 0.1
    grad_clip:           float = 1.0

@dataclass
class SystemConfig:
    bandwidth_mbps:       List[int] = field(default_factory=lambda: [1, 10, 100])
    protocol_overhead_ms: float     = 2.0
    edge_memory_limit_mb: int       = 4096
    edge_compute_factor:  float     = 8.0

@dataclass
class AllParams:
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    data:     DatasetConfig  = field(default_factory=DatasetConfig)
    train:    TrainingConfig = field(default_factory=TrainingConfig)
    system:   SystemConfig   = field(default_factory=SystemConfig)
    models:   Dict = field(default_factory=lambda: MODEL_ZOO)
    comps:    Dict = field(default_factory=lambda: COMPENSATOR_CONFIG)
    colors:   Dict = field(default_factory=lambda: COLORS)

# ── 全局实例 ──────────────────────────────────────────────────────────
paras = AllParams()


if __name__ == "__main__":
    from Configs.paras import paras
    from dataclasses import replace
    
    # 直接用全局实例   
    print(paras.train.lr)          # 0.001
    print(paras.hardware.device)   # "cuda"

    # 临时改某个参数（不影响全局）
    local_paras = replace(paras, train=replace(paras.train, lr=1e-4, epochs=10))
    print(local_paras.train.lr)   # 1e-4
    print(paras.train.lr)         # 0.001，全局没动

    # 单独 import
    from Configs.paras import TrainingConfig, MODEL_ZOO, COMPENSATOR_CONFIG
    cfg = TrainingConfig(lr=1e-4, epochs=20)   # 自定义实例
    model_info = MODEL_ZOO["resnet50"]
    print(cfg)
    print(model_info)

    # 访问 dict 类配置
    for name, cfg in paras.comps.items():
        if cfg["enabled"]:
            print(f"启用补偿器: {name}")
    color = paras.colors["blue"]   # '#1f77b4'
    print(color)