#!/bin/bash

export PYTHONPATH=${PWD} 
export CUDA_VISIBLE_DEVICES=3 

python robosac.py \
    --log \
    --robosac no_defense \
    --adv_method fgsm \
    --number_of_attackers 1 \
    --random_attack
    # --visualization



# 注意，生成数据的时候不要开启 visualization