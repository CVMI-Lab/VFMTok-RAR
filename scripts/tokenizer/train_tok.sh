# !/bin/bash
mkdir -p output/logs
export NODE_COUNT=$1
export NODE_RANK=$2
export PROC_PER_NODE=1
export MASTER_PORT=22333
export MASTER_ADDR='localhost'
mkdir -p output/logs
rm -rf titok/__pycache__ paintmind/data/__pycache__ vfmtok/tokenizer/__pycache__
scripts/autoregressive/torchrun.sh vq_train.py  --image-size 336 --results-dir output --mixed-precision none --codebook-slots-embed-dim 12    \
    --data-path imagenet/lmdb/train_lmdb --global-batch-size 8 --num-workers 4 --ckpt-every 5000 --epochs 50 \
    --transformer-config configs/vit_transformer.yaml --log-every 1 --lr 1e-4 --ema --z-channels 512 \
    2>&1 | tee 'train_tok.log'