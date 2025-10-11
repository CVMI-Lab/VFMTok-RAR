# !/bin/bash
export NODE_COUNT=1
export NODE_RANK=0
export PROC_PER_NODE=8
eval $(curl -s http://deploy.i.shaipower.com/httpproxy)
iters="checkpoint-$(printf "%06d" "$1")"
rm -rf vfmtok/titok/__pycache__ vfmtok/data/__pycache__ vfmtok/core/__pycache__
scripts/autoregressive/torchrun.sh test_net.py --config-file configs/training/generator/rar.yaml --compile \
     --gpt-ckpt snapshot/RAR-L/${iters}/model.safetensors --image-size 256 --image-size-eval 256 --per-proc-batch-size $2 \
     --guidance-scale $3 --sample-dir samples --guidance-scale-pow 1 2>&1 | tee 'hello.log'