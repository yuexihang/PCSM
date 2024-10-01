data_root=${1}

python exp_plasticity.py \
    --gpu 0 \
    --n-hidden 128 \
    --n-heads 8 \
    --n-layers 8 \
    --lr 0.001 \
    --max_grad_norm 0.1 \
    --batch-size 8 \
    --freq_num 128 \
    --unified_pos 0 \
    --ref 8 \
    --eval 0 \
    --data_path ${data_root} \
    --ntrain 900 \
    --save_name Plas_PCSM
