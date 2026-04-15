# nohup Scripts/Exp2_Compensator/run_benchmark.sh < /dev/null > /home/ResidualRemove/Results/Exp2_Compensator/nohup$(date +%Y%m%d_%H%M).log 2>&1 &

# 正式测试，使用更多校准图像和轮数，确保结果稳定可靠
python -m Scripts.Exp2_Compensator.run_benchmark \
    --model resnet50 \
    --dataset imagenet \
    --device cuda:0 \
    --batch-size 64 \
    --num-workers 4 \
    --calib-size 4196 \
    --epochs 10 \
    --removed-blocks "layer4.0" \
    --pretrained \
    --latency-reps 50 \
    --latency-warmup 10

# 快速测试，减少校准图像数量和轮数，限制最大批次数，确保在合理时间内完成
python -m Scripts.Exp2_Compensator.run_benchmark \
    --model resnet50 \
    --dataset imagenet \
    --device cpu \
    --batch-size 64 \
    --num-workers 4 \
    --calib-size 512 \
    --epochs 1 \
    --removed-blocks "layer4.2" \
    --pretrained \
    --max-batches 10