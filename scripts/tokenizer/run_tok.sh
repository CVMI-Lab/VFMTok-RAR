# !/bin/bash
export NODE_COUNT=1
export NODE_RANK=0
export PROC_PER_NODE=8
export MASTER_PORT=21345
rm -rf engine/__pycache__ tokenizer/tokenizer_image/__pycache__
scripts/autoregressive/torchrun.sh vqgan_test.py --vq-model VQ-16 --image-size 336 --output_dir recons --batch-size $2   \
        --z-channels 512 --codebook-slots-embed-dim 12 --vq-ckpt tokenizer/vit_vqgan_step_$1.pt 2>&1 | tee 'test.log'