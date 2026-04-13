# 动机实验
- **==基础配置==**
	- **模型**：ResNet、MobileNetV2
	- **数据**：CIFAR、ImageNet
- **==实验1：动机实验==**
	- ==删除的好处==
		1. **Exp1-1：针对资源受限终端**
			- 设备的峰值**内存占用**（下降）
			- 特征张量的存活时间（缩短）
			- 整体**推理延迟**（降低）、吞吐量（提升）
			- 设备的 DRAM **访存能耗**（降低）（暂不实验，实现比较复杂）
				在边缘设备上，搬运数据（Memory Access）的能耗远大于计算（MACs）。残差的存在迫使张量写回主存再读出，证明删残差能显著降低设备功耗（这对电池供电的端侧设备是致命痛点）。
		2. **Exp1-2：针对端边云协同的模型切分传输**
			- 数据量（降低）
				- 可以配图对比 DAG拓扑 和 链式拓扑 切分点两条边的数据量
			- 特征图序列化时间开销（降低）
			- 带宽开销（降低）
				- 可以模拟不同的网络带宽环境在切分点跨设备时，额外数据传输量如何让端到端延迟恶化
	- ==进行后训练补偿的必要性==
		1. **Exp1-3：直接删除会导致精度下降**
			逐层删除残差后模型的 top-1 acc 下降趋势（证明直接删会导致原始特征被破坏）
		2. **Exp1-4：残差是有结构的信号**
			每个 block $x$ 和 $\mathcal{F}(x,W)$ 的 L2-norm 比值、余弦相似度
		3. 分析浅层 vs 深层 block 的差异（有时间再做）

---

# 核心方法验证
- **==基础配置==**
	- **模型**：ResNet-50
	- **数据**：ImageNet
	- **策略**：选择统一的后训练校准策略，冻结主干参数 $W$，使用 1024 张校准图像，迭代微调（或解析求解）新增参数 $W_{\text{new}}$
- **==实验2：补偿方法基准测试对比==**
	- **对比对象**：Scalar / Affine / Linear 1×1 / Low-rank (不同 r) / Adapter
	- **对比指标**：Acc、参数量、FLOPS、峰值内存、推理延迟、~~能耗~~ 的变化

| **方法**                 | **公式**                   | **作用与系统特性**                          |
| ---------------------- | ------------------------ | ------------------------------------ |
| 标量 (Scalar)            | $\alpha \cdot z$         | Baseline，拟合能力极弱。                     |
| 仿射 (Affine)            | $\gamma \odot z + \beta$ | Baseline，部署时可与前置 Conv/BN 算子融合（零开销）   |
| 映射 (Linear $1\times1$) | $z+W_{1\times1} \cdot z$ | **核心方案1**，表达力尚可，**可通过重参数化收入主干**（零开销） |
| 低秩 (LoRA)              | $z+W_2W_1z$              | **核心方案2**，极少参数量（微小矩阵乘开销）             |
| 适配器 (Adapter)          | $z+W_2\sigma(W_1 z)$     | Upper Bound，非线性表达力强，但引入额外计算图（违背初衷）   |

- **==实验3：端到端协同推理性能评估==**
	- **设置**：根据建模，部署在真实的端侧设备（如 Raspberry Pi, Jetson Nano, 或 Android 手机）和云端服务器。设定网络切分点，中间通过网络传输中间特征
	- **变量**：
		1. 原始 ResNet（带残差，传输特征 $z$ + $x$）
	    2. 改造后的纯链式 ResNet（仅传输特征 $z$）
	    3. 网络带宽（模拟不同信道质量）
	- **指标**：
	    - **端到端延迟**：Edge 计算 + 传输 + Cloud 计算的总耗时
		    画堆叠柱状图，将时间分解为：`端侧计算时间` + `网络传输时间` + `云端计算时间`。
	    - **传输数据量**：切分点处需要传输的 tensor size（有残差时块内切分要传两路，链式只需一路）
	    - **Peak Memory**（Edge 侧和 Cloud 侧）：边缘设备内存是瓶颈
	    - **Top-1 Acc**：用于对比精度代价
	- **形式**：X 轴是不同带宽和模型（1Mbps, 10Mbps, 100Mbps），Y 轴是对应模型在对应带宽下的端到端延迟的堆叠柱状图（端侧耗时 + 传输耗时 + 云侧耗时）、数据量、显存占用、精度损失对比
	- **目的**：证明在网络带宽有限的情况下，省去 $x$ 的传输能让 `网络传输时间` 大幅缩短，从而获得极佳的端到端加速比。
	
- **==实验4：边缘设备的多租户/高并发吞吐量==**
	- **逻辑**：删除残差后，峰值内存下降，那么同样的边缘 GPU/NPU 显存，就能塞下更大的 Batch Size，或者同时跑更多的模型实例。
	- **指标**：并发请求下的系统吞吐量（Queries Per Second, QPS）和显存占用峰值。
	- **目的**：证明方法不仅是加速单次推理，而是**提升了整个边缘节点的资源利用率和系统容量**。

---

# 其他实验
- **==实验5：模型泛化性==**
    - 在 MobileNetV2、EfficientNet、MobileNetV3 上重跑
    - 目的：说明方法不是 ResNet-specific 的
- **==实验6：校准数据量消融==**
    - 用最终方案，变化校准数据量（128 / 512 / 1k / 5k 张）
    - 目的：说明方法对数据量不敏感，适合部署场景
- **==实验7：局部移除策略==**
	- 对比原始残差网络 vs 链式网络，统计在不同切分位置下，切分点数量（即有效的切分粒度）
	- 对比“全量删除并补偿”与“仅删除切分点边界的 1~2 个残差块并补偿” 
	- 目的：量化说明链式拓扑使切分策略空间更大，给端边云的负载均衡和动态调度提供更多自由度
		- Trade-off （保留最大精度，去除最大瓶颈）

---

# 实验架构设计
```textile
ResidualRemoval/
│
├── Configs/                   # 【静态配置文件】
│   ├── compensator.yaml       
│   ├── default_env.yaml       
│   ├── models.yaml            
│   └── system.yaml            
│
├── Results/                   # 【最终输出结果】
│   ├── Exp1_Motivation/       
│   │   ├── Motivation1_Inference_cost/
│   │   ├── Motivation2_Collaborate_cost/
│   │   ├── Motivation3_Acc_drop/
│   │   └── Motivation4_Residual_stats/
│   └── Exp2_Compensator/      
│
├── Scripts/                   # 【实验与运行脚本】
│   ├── Exp1_Motivation/       # 动机实验
│   │   ├── run1_inference_cost.py & .sh
│   │   ├── run2_collaborate_cost.py & .sh
│   │   ├── run3_acc_drop.py & .sh
│   │   └── run4_residual_stats.py & .sh
│   ├── Exp2_Compensator/      
│   │   └── run_benchmark.py & .sh
│   ├── Exp3_System/           
│   │   ├── run_concurrent.py
│   │   └── run_split_inference.py
│   ├── Exp4_Ablation/         
│   │   ├── run_calib_size.py
│   │   ├── run_generalization.py
│   │   └── run_partial_removal.py
│   ├── Tests/                 # 单元测试与模块验证
│   │   └── test_models.py
│   ├── __init__.py
│   └── common.py              # 脚本公共组件
│
├── Src/                       # 【核心源代码】
│   ├── Collab_System/         # 端边协同系统相关
│   │   ├── bandwidth_sim.py
│   │   ├── split_runner.py
│   │   └── tensor_transfer.py
│   ├── Models_Evaluation/     # 模型评估模块
│   │   ├── accuracy.py
│   │   ├── flops.py
│   │   ├── latency.py
│   │   └── memory.py
│   ├── Models_Nets/           # models
│   │   ├── origin/            # 原始未经修改的 Backbones
│   │   │   ├── mobilenet.py
│   │   │   └── resnet.py
│   │   ├── builder.py
│   │   ├── compensators.py
│   │   └── injector.py        # 负责网络改写与旁路注入
│   ├── Models_Training/       # 训练与校准
│   │   ├── calibrate.py
│   │   ├── loss.py
│   │   └── trainer.py
│   ├── Plots/                 # 可视化绘图
│   │   ├── plot_acc_drop.py
│   │   ├── plot_compensator_table.py
│   │   ├── plot_residual_stats.py
│   │   └── plot_split_latency.py
│   └── Utils/                 # 工具类
│       ├── calibration.py
│       ├── config.py          
│       ├── datasets.py
│       └── runtime.py         
│
└── .gitignore                
```