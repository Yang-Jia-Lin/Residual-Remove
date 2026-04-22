## 数据解压
```bash
cd /root/autodl-tmp/0-Data/ImageNet100
unzip imagenet100.zip
```
先确认解压后的结构：

```bash
ls /root/autodl-tmp/0-Data/ImageNet100/imagenet100/ | head -5
ls /root/autodl-tmp/0-Data/ImageNet100/imagenet100/ | wc -l  # 应该是100个类
```

## 使用流程

**第一步：生成索引文件（只需运行一次）**
```bash
python split_imagenet100.py
```
会在 `/root/autodl-tmp/0-Data/ImageNet100/split_indices.json` 生成切分索引。

**第二步：验证数据泄漏**
```bash
python split_imagenet100.py --verify
```

**第三步：训练代码里 import**
```python
from split_imagenet100 import get_datasets

train_ds, val_ds = get_datasets(
    train_transform=train_transform,
    val_transform=val_transform
)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)
```
