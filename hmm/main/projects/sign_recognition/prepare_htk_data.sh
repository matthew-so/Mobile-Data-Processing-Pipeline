#!/bin/sh

python3 driver.py \
    --test_type none \
    --features_file configs/features_sign.json \
    --data_path data/data_fs \
    --parallel_jobs 32 \
    --is_fingerspelling \
    --is_bigram \
    --prepare_data \
