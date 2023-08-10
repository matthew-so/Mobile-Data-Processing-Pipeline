#!/bin/sh
# --wordlist signs.txt \

python3 driver.py \
    --test_type test_on_train \
    --features_file configs/features_sign.json \
    --prototypes_file configs/prototypes.json \
    --data_path $1 \
    --train_iters 150 \
    --n_states 8 \
    --parallel_jobs 32 \
    --random_state 43556 \
    --hmm_insertion_penalty -200
