export PYTHONPATH=${PWD}
CUDA_VISIBLE_DEVICES=3 

python robosac.py \
    --log \
    --robosac no_defense \
    --adv_method GN \
    --visualization \
    # --adv_iter 15 \   # Default: 15
    # --number_of_attackers 1 \
    # --ego_agent 1 \

