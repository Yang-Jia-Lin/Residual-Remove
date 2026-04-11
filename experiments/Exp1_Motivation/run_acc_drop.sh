# nohup /home/ResidualRemove/experiments/Exp1_Motivation/run_acc_drop.sh > _logs/exp1_output.log 2>&1 &
# 调试时可以加 --max-batches 20 --val-size 1000 快速验证流程是否正确
python experiments/Exp1_Motivation/run_acc_drop.py \
    --model resnet50 \
    --dataset ImageNet \
    --device cpu \
    --batch-size 128 \
    --num-workers 8 \
    --pretrained \
    --max-batches 20 \
    --val-size 1000