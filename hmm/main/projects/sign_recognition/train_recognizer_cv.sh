#!/bin/sh

python3 driver.py \
    --test_type cross_val \
    --features_file configs/features_sign.json \
    --prototypes_file configs/prototypes.json \
    --data_path $1 \
    --train_iters 30 \
    --hmm_insertion_penalty -200 \
    --cv_parallel \
    --n_splits 5 \
    --parallel_jobs 32 \
    --cross_val_method kfold
