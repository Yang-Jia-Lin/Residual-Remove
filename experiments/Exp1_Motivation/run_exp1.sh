# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name
python experiments/Exp1_Motivation/run_acc_drop.py \
    --model resnet50 \
    --dataset imagenet \
    --device cuda:0 \
    --batch-size 128 \
    --num-workers 8 \
    --pretrained