# nohup Scripts/Exp2_Compensator/run_benchmark.sh < /dev/null > /home/ResidualRemove/Results/Exp2_Compensator/nohup$(date +%Y%m%d_%H%M).log 2>&1 &

python -m Scripts.Exp2_Compensator.run_benchmark \
    --model resnet50 \
    --dataset imagenet \
    --device cpu \
    --batch-size 64 \
    --num-workers 4 \
    --pretrained \
    --calib-size 1024 \
    --epochs 3