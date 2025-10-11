# !/bin/bash
export NCCL_IB_HCA=$(pushd /sys/class/infiniband/ > /dev/null; for i in mlx*_*; do cat $i/ports/1/gid_attrs/types/* 2>/dev/null | grep v >/dev/null && echo $i ; done; popd > /dev/null)
echo ${NCCL_IB_HCA}
export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=5
export NCCL_IB_HCA=mlx5_12
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_IB_TC=186
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_TIMEOUT=18
export NCCL_IB_RETRY_CNT=7
rm -rf modeling/__pycache__ paintmind/modules/encoders/__pycache__ ./__pycache__ modeling/__pycache__
eval $(curl -s http://deploy.i.shaipower.com/httpproxy)
rm -rf paintmind/titok/__pycache__ paintmind/data/__pycache__ tokenizer/vfmtok/__pycache__ utils/__pycache__ utils/__pycache__
accelerate launch --config_file $1 train_rar.py --config-file configs/training/generator/rar.yaml  \
    --image-size 336 --anno-file imagenet/lmdb/train_lmdb --num-workers 4 2>&1 | tee 'train.log'