#!/bin/bash

export PYTHONPATH=${PWD} 
export CUDA_VISIBLE_DEVICES=3 

nohup python robosac.py \
    --log \
    --robosac no_defense \
    --adv_method fgsm \
    --random_attack \
    --number_of_attackers 2 &

