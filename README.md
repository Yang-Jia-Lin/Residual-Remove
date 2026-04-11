# ResidualRemove

ResidualRemove 是一个围绕“删除残差连接并用轻量补偿器恢复精度”的初版实验代码库。当前版本优先完成实验骨架、统一配置、可运行脚本和基本分析工具，方便后续替换成真实预训练权重、ImageNet/CIFAR 数据和设备测量结果。

## 当前范围

- 已建立与你的实验设计对应的目录结构。
- 已提供 ResNet 和 MobileNetV2 的可切换残差/补偿器实现骨架。
- 已提供校准、评估、系统模拟、可视化和 4 组实验入口脚本。
- 默认支持 `synthetic` 回退模式，在没有真实数据和权重时也可以跑通 smoke test。

## 安装

```bash
pip install -r requirements.txt
```

## 推荐起步命令

```bash
python experiments/Exp1_Motivation/run_acc_drop.py --device cpu --dataset synthetic
python experiments/Exp1_Motivation/run_residual_stats.py --device cpu --dataset synthetic
python experiments/Exp2_Compensator/run_benchmark.py --device cpu --dataset synthetic --epochs 1 --calib-size 32
python experiments/Exp3_System/run_split_inference.py --device cpu --dataset synthetic
python experiments/Exp4_Ablation/run_partial_removal.py --device cpu --dataset synthetic
```

## 关键约定

- `configs/default_env.yaml` 管理数据根目录、结果目录、随机种子和默认设备。
- `configs/models.yaml` 管理模型基础超参。
- `configs/compensator.yaml` 管理校准训练和补偿器参数。
- `configs/system.yaml` 管理带宽、能耗和内存预算假设。
- 所有实验脚本都会把结果写入 `results/` 下对应子目录。

## 当前限制

- 没有附带预训练权重，因此精度结果主要用于验证 pipeline，而不是最终论文指标。
- `split_runner` 当前按 block 边界模拟切分，适合作为初版系统实验，不是最终部署框架。
- 能耗、吞吐量与并发分析目前是 profile + analytic simulation 的组合，后续可以替换为真实设备测量。
