#!/usr/bin/env bash
source ~/.bashrc

module load cuda/10.2

conda activate mmaction

CONFIG=/home/abali/mmaction2/work_dir/config_slowfast_varuna.py  #modify path to configuration file
GPUS=2
PORT=${PORT:-29500}
VALIDATE=True

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
       $(dirname "$0")/tools/train.py $CONFIG --validate --launcher pytorch ${@:3}

conda deactivate

