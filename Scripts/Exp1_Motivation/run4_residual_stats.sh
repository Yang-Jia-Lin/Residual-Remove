# nohup Scripts/Exp1_Motivation/run4_residual_stats.sh < /dev/null > Results/Exp1_Motivation/Motivation4_Residual_stats/log_$(date +%Y%m%d_%H%M).log 2>&1 &

python -m Scripts.Exp1_Motivation.run4_residual_stats \
    --model resnet50 \
    --dataset imagenet \
    --device cpu \
    --batch-size 64 \
    --num-workers 4 \
    --pretrained \
    --max-batches 20

# 调试时加 --max-batches 20 快速验证流程。