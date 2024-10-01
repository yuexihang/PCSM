data_root=${1}

python exp_ns.py \
    --n-hidden 256 \
    --n-heads 8 \
    --n-layers 8 \
    --lr 0.001 \
    --batch-size 4 \
    --freq_num 128 \
    --unified_pos 1 \
    --ref 8 \
    --eval 0 \
    --data_path ${data_root} \
    --ntrain 1000 \
    --save_name NS_PCSM
