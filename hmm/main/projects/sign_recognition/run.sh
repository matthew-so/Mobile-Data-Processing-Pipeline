#!/bin/sh

python3 driver.py \
    --test_type cross_val \
    --cross_val_method stratified \
    --train_iters 150 \
    --n_splits 10 \
    --hmm_insertion_penalty -200 \
    --cv_parallel \
    --parallel_jobs 8
