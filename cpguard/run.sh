export CUDA_VISIBLE_DEVICES=2

nohup python train.py --CADet > CADet.log 2>&1 &