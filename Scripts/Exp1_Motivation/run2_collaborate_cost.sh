# nohup Scripts/Exp1_Motivation/run2_collaborate_cost.sh < /dev/null > Results/Exp1_Motivation/Motivation2_Collaborate_cost/log_$(date +%Y%m%d_%H%M).log 2>&1 &

python -m Scripts.Exp1_Motivation.run2_collaborate_cost \
    --model resnet50 \
    --dataset imagenet \
    --device cpu \
    --batch-size 1 \
    --num-workers 4 \
    --pretrained
    
# batch-size 建议设为 1，因为这里关心的是单张图片的张量大小，而不是 batch 级别的吞吐量。