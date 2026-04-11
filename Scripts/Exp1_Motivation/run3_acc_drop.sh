# nohup Scripts/Exp1_Motivation/run3_acc_drop.sh < /dev/null > Results/Exp1_Motivation/Motivation3_Acc_drop/log_$(date +%Y%m%d_%H%M).log 2>&1 &

python -m Scripts.Exp1_Motivation.run3_acc_drop \
    --model resnet50 \
    --dataset imagenet \
    --device cpu \
    --batch-size 128 \
    --num-workers 8 \
    --pretrained \
    --max-batches 20 \
    --val-size 1000
    
# 调试时可以加 --max-batches 20 --val-size 1000 快速验证流程是否正确