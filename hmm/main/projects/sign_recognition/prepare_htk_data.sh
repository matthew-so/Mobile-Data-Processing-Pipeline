#!/bin/sh

python3 driver.py \
    --test_type none \
    --features_file configs/features_sign.json \
    --parallel_jobs 32 \
    --is_single_word \
    --prepare_data \
