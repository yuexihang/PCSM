data_root=${1}
python exp_darcy.py \
    --n-hidden 128 \
    --n-heads 8 \
    --n-layers 8 \
    --lr 0.001 \
    --max_grad_norm 0.1 \
    --batch-size 4 \
    --freq_num 128 \
    --unified_pos 1 \
    --ref 8 \
    --eval 0 \
    --downsample 5 \
    --data_path ${data_root} \
    --ntrain 1000 \
    --save_name Darcy_PCSM
