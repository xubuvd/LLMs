#!/bin/bash

train_file=./data/clean_data.txt.train

python train.py \
    --input_file $train_file \
    --pretrain_w2v ./w2v/wiki-news-300d-1M.vec \
    --model_file ./output_models/imagenet.bin \
    --stop_word_path ./data/stop_words_en.txt \
    --output_path ./data/imagenet_res.jsonl \
    --mode train
if [ $? -ne 0 ]; then
    echo "train.py: ${train_file} failed."
    exit
else
    echo "train.py: ${train_file} succeed."
fi

