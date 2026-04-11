# nohup /home/ResidualRemove/experiments/Exp1_Motivation/run_system_cost.sh > _logs/exp1-2_output.log 2>&1 &

python experiments/Exp1_Motivation/run_system_cost.py \
    --model resnet50 \
    --dataset imagenet \
    --device cpu \
    --batch-size 1 \
    --num-workers 4 \
    --pretrained
    
# batch-size 建议设为 1，因为这里关心的是单张图片的张量大小，而不是 batch 级别的吞吐量。