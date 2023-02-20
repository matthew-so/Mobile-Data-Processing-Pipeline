#!/bin/sh

python3 driver.py \
    --test_type test_on_train \
    --features_file configs/features_sign.json \
    --train_iters 60 \
    --hmm_insertion_penalty -200
