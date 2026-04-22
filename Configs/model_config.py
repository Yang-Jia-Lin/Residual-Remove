"""Configs/model_config.py"""
from dataclasses import dataclass, field


@dataclass
class HardwareConfig:
    seed:        int  = 42
    device:      str  = "cuda"
    num_workers: int  = 4
    pin_memory:  bool = True

@dataclass
class DatasetConfig:
    default_dataset:      str  = "imagenet100"
    default_image_size:   int  = 224
    calib_num_samples:    int  = 2048
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
class AllModelConfig:
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    data:     DatasetConfig  = field(default_factory=DatasetConfig)
    train:    TrainingConfig = field(default_factory=TrainingConfig)


model_config = AllModelConfig()