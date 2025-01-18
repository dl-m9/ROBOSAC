#!/bin/bash

export PYTHONPATH=${PWD} 
export CUDA_VISIBLE_DEVICES=0





eps=0.25
adv_method="fgsm"
number_of_attackers=2

# 注意，生成数据的时候不要开启visualization.


# for adv_method in $adv_methods; do
nohup python robosac.py \
    --log \
    --robosac cp-guard \
    --adv_method $adv_method \
    --number_of_attackers $number_of_attackers \
    --data /data2/user2/senkang/CP-GuardBench/V2X-Sim-det/test_8/ \
    --eps $eps  \
    --ego_agent 1 > "NoDefense_${adv_method}_${eps}_${number_of_attackers}.out" 2>&1 &
        # --visualization
# done
# nohup python robosac.py \
#         --log \
#         --robosac no_defense \
#         --number_of_attackers $number_of_attackers \
#         --data /data2/user2/senkang/CP-GuardBench/V2X-Sim-det/train/ \
#         --random_attack \
#         --ego_agent 1 > "generate2.out" 2>&1 &
#         # --visualization