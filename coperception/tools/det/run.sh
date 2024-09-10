#!/bin/bash

export PYTHONPATH=${PWD} 
export CUDA_VISIBLE_DEVICES=2

python robosac.py \
    --log \
    --robosac no_defense \
    --adv_method cw-l2 \
    --number_of_attackers 1 \
    --random_attack \
    --data /data2/user2/senkang/CP-GuardBench/V2X-Sim-det/train/ 
    # --visualization



# 注意，生成数据的时候不要开启 c