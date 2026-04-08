# ResidualRemove

用于做 ResNet 残差连接删除实验的最小脚手架，目标是比较不同残差保留比例下的：

- Top-1 Accuracy
- Peak Memory
- Activation Lifetime Proxy
- Latency / Throughput

## 项目结构

```text
ResidualRemove/
├─ configs/
│  └─ ablation_presets.json
├─ results/
├─ scripts/
│  ├─ run_ablation_suite.py
│  └─ train_eval.py
├─ src/
│  └─ residual_remove/
│     ├─ models/
│     │  └─ resnet_ablation.py
│     └─ utils/
│        ├─ data.py
│        ├─ metrics.py
│        └─ training.py
└─ requirements.txt
```

## 安装

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## 单次训练 + 评测

```powershell
.\.venv\Scripts\python scripts\train_eval.py `
  --model resnet18 `
  --dataset cifar10 `
  --data-root .\data `
  --epochs 1 `
  --batch-size 64 `
  --ablation-mode random_ratio `
  --ablation-value 0.5 `
  --output-dir .\results\resnet18_half
```

## 批量跑预设实验

```powershell
.\.venv\Scripts\python scripts\run_ablation_suite.py `
  --dataset cifar10 `
  --data-root .\data `
  --epochs 1 `
  --batch-size 64 `
  --output-root .\results
```

## 说明

- `random_ratio`：随机删除指定比例的残差连接。
- `stage_progressive`：按 stage 删除，例如传入 `1` 表示删除 `layer1`，传入 `3` 表示删除 `layer1~layer3`。
- `full`：删除全部残差连接。
- `Activation Lifetime Proxy` 不是底层显存分配器级别的真实生命周期，而是一个针对残差缓存张量的近似代理指标，用于横向比较不同 ablation 方案。
