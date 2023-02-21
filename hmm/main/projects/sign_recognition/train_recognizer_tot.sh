#!/bin/sh

python3 driver.py \
    --test_type test_on_train \
    --features_file configs/features_sign.json \
    --prototypes_file configs/prototypes.json \
    --train_iters 30 \
    --hmm_insertion_penalty -200