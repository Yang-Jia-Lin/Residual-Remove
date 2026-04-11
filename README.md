# ResidualRemove

ResidualRemove 是一个围绕“删除残差连接，并用轻量补偿器恢复精度”的实验代码库。仓库当前已经具备模型构建、后训练校准、系统级模拟、统计分析和可视化所需的主要脚手架，适合用来复现实验流程、生成烟雾测试结果，并逐步替换为真实数据、真实权重和真实设备测量。

## 项目组织结构

当前仓库按“配置、数据、实验、模型、评估、训练、系统、可视化”分层组织：

- `_configs/`：全局配置文件
	- `default_env.yaml`：数据根目录、输出根目录、默认设备、随机种子、是否启用 synthetic 回退
	- `models.yaml`：模型基础参数，如 ResNet-18/50、MobileNetV2 的输入尺寸与类别数
	- `compensator.yaml`：补偿器训练超参、校准集大小、学习率等
	- `system.yaml`：带宽、协议开销、内存预算、DRAM 与 MAC 能耗假设
- `_data/`：数据集目录
	- `CIFAR10/`
	- `ImageNet/`
- `_logs/`：训练与运行日志、检查点
	- `checkpoints/`
	- `terminal_logs/`
- `_results/`：已有的 smoke 结果与示例输出
	- `motivation/`
	- `compensator/`
	- `system/`
	- `ablation/`
- `experiments/`：实验入口脚本
	- `common.py`：共享参数、配置加载、数据集和模型构建
	- `Exp1_Motivation/`：动机实验
	- `Exp2_Compensator/`：补偿方法基准测试
	- `Exp3_System/`：端到端系统协同推理与并发吞吐量评估
	- `Exp4_Ablation/`：泛化、校准集大小、局部移除策略消融
- `models/`：模型与补偿器实现
	- `resnet.py`：ResNet 残差块与残差/纯链式/补偿式前向模式
	- `mobilenet.py`：MobileNetV2 残差块实现
	- `compensators.py`：Scalar、Affine、Linear 1x1、Low-rank、Adapter 等补偿器
	- `builder.py`：统一模型构建入口
- `src/`：训练、评估、系统模拟和可视化工具
	- `evaluation/`：accuracy、latency、memory、flops、energy
	- `system/`：带宽模拟、切分推理、张量传输
	- `training/`：校准训练和评估
	- `utils/`：配置、数据集、运行时工具
	- `visualization/`：结果绘图脚本

## 实验设计

下面的实验方案与现有脚本一一对应。当前代码默认提供 smoke 配置和 synthetic 回退；论文级实验建议在补齐真实数据、预训练权重和设备环境后再跑最终结果。

### 1. 动机实验

目标是说明残差连接“为什么要删”“为什么不能直接删”，并量化它在内存、延迟、带宽和能耗上的代价。

- 基础模型：ResNet-18、ResNet-50、MobileNetV2
- 建议数据集：CIFAR-100、ImageNet
- 代码入口：
	- `experiments/Exp1_Motivation/run_acc_drop.py`：逐步删除残差块，统计 Top-1 / Top-5 下降趋势
	- `experiments/Exp1_Motivation/run_residual_stats.py`：统计每个残差块的 L2 范数比值与余弦相似度
	- `experiments/Exp1_Motivation/run_system_cost.py`：模拟 DAG 拓扑与链式拓扑在不同带宽下的传输代价

建议观察的指标包括：峰值内存占用、特征张量存活时间、端侧推理延迟、吞吐量、DRAM 访存能耗、跨设备传输字节数和端到端时延。

### 2. 核心方法验证

目标是验证补偿器是否能在“冻结主干、仅训练新增参数”的前提下，尽量恢复删除残差后的精度，同时保持系统开销最小。

- 基础模型：ResNet-50
- 建议数据集：ImageNet
- 统一策略：后训练校准，冻结主干，仅更新补偿器参数，默认使用 1024 张校准图像进行微调或解析求解
- 代码入口：
	- `experiments/Exp2_Compensator/run_benchmark.py`：比较 Scalar、Affine、Linear 1x1、Low-rank、Adapter 等补偿器
	- `experiments/Exp3_System/run_split_inference.py`：评估切分推理下的端到端时延与传输开销
	- `experiments/Exp3_System/run_concurrent.py`：评估显存约束下的多租户吞吐量和最大并发数

重点对比指标包括：Top-1 Accuracy、参数量变化、推理延迟变化、MACs/FLOPs 变化、峰值显存和估算能耗。Scalar 和 Affine 可作为零或近零开销基线，Linear 1x1 是主要候选方案，Low-rank 是参数更省的替代，Adapter 则作为表达能力上界。

### 3. 其他实验

目标是验证方法的泛化性、对校准数据量的鲁棒性，以及局部移除策略对切分空间的影响。

- 代码入口：
	- `experiments/Exp4_Ablation/run_generalization.py`：在 MobileNetV2 等模型上验证泛化性
	- `experiments/Exp4_Ablation/run_calib_size.py`：做校准集大小消融
	- `experiments/Exp4_Ablation/run_partial_removal.py`：比较全量删除与局部边界删除策略

这部分实验主要用来说明方法不是 ResNet 专属，也不依赖极大的校准集，同时链式拓扑会扩大切分粒度，给端边云调度更多自由度。

## 默认支持的模型与数据

- 模型：ResNet-18、ResNet-50、MobileNetV2
- 数据集：CIFAR-10、CIFAR-100、ImageNet
- 回退模式：当真实数据缺失时，`src/utils/datasets.py` 会自动切换到 synthetic 数据，保证脚本可以跑通 smoke test

## 安装

		pip install -r requirements.txt

## 快速开始

如果只想先确认管线是否可运行，可以直接跑 synthetic smoke test。

		python experiments/Exp1_Motivation/run_acc_drop.py --device cpu --dataset synthetic
		python experiments/Exp1_Motivation/run_residual_stats.py --device cpu --dataset synthetic
		python experiments/Exp1_Motivation/run_system_cost.py --device cpu --dataset synthetic
		python experiments/Exp2_Compensator/run_benchmark.py --device cpu --dataset synthetic --epochs 1 --calib-size 32
		python experiments/Exp3_System/run_split_inference.py --device cpu --dataset synthetic
		python experiments/Exp3_System/run_concurrent.py --device cpu --dataset synthetic
		python experiments/Exp4_Ablation/run_generalization.py --device cpu --dataset synthetic
		python experiments/Exp4_Ablation/run_calib_size.py --device cpu --dataset synthetic
		python experiments/Exp4_Ablation/run_partial_removal.py --device cpu --dataset synthetic

## 配置说明

- `default_env.yaml` 负责环境与路径约定
- `models.yaml` 负责模型尺度和类别数
- `compensator.yaml` 负责补偿器校准超参
- `system.yaml` 负责带宽、内存和能耗建模

如果你要切换到真实数据，只需要先把数据按 `_data/CIFAR10/`、`_data/CIFAR100/`、`_data/ImageNet/train` 和 `_data/ImageNet/val` 放好，再在命令行里通过 `--dataset` 和 `--model` 选择对应组合即可。

## 输出与可视化

- 实验结果默认写入 `results/` 目录树下的 CSV 文件
- 仓库中已有的示例 smoke 输出保存在 `_results/`
- 绘图脚本位于 `src/visualization/`，可直接基于 CSV 结果生成论文图表

## 当前限制

- 当前仓库以实验管线和可复现 smoke test 为主，不包含最终论文所需的完整预训练权重
- 切分推理和带宽分析目前是解析建模与 profile 的组合，真实端侧设备部署仍需额外接入
- 若要对齐最终实验，请使用真实的 CIFAR-100、ImageNet 数据，以及你计划报告的硬件环境
