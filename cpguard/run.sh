export CUDA_VISIBLE_DEVICES=2

attack_type="fgsm bim cw-l2"

for attack in $attack_type; do
    nohup python train.py --msc_loss --leave_one_out $attack > MSC_leave_one_out_$attack.txt 2>&1 &
done
