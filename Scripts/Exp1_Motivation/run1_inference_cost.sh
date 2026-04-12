# nohup Scripts/Exp1_Motivation/run1_inference_cost.sh < /dev/null > Results/Exp1_Motivation/Motivation1_Inference_cost/$(date +%Y%m%d_%H%M).log 2>&1 &
    
python -m Scripts.Exp1_Motivation.run1_inference_cost \
    --model resnet50 \
    --dataset imagenet \
    --device cuda:0 \
    --batch-size 64 \
    --num-workers 4 \
    --pretrained

# 内存测试建议用较大的 batch（如 32 或 64），因为 batch 越大，残差分支 x 的内存占用越显著，节省效果越明显。
# batch=1 时 x 的大小相对于模型权重本身可以忽略不计，体现不出显存压力的实质差异。