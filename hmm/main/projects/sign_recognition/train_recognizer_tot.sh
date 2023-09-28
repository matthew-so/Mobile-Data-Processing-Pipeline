#!/bin/sh
# --wordlist signs.txt \

python3 driver.py \
    --test_type test_on_train \
    --features_file configs/features_sign.json \
    --prototypes_file configs/prototypes.json \
    --data_path $1 \
    --wordList $1/wordList_letter \
    --hBuildWordList $1/wordList_word \
    --dict $1/dict \
    --grammar_file $1/grammar.txt \
    --all_labels_file $1/all_labels_letter.mlf \
    --is_bigram \
    --is_fingerspelling \
    --train_iters 5 \
    --n_states 3 \
    --parallel_jobs 32 \
    --random_state 43556 \
    --hmm_insertion_penalty -200
