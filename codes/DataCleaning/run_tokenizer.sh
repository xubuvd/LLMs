#!/bin/bash

dataset="Alpaca52k"
clearning_version="v1"

src_dir="/Users/mac/Downloads/sft/paper/sft_train4"
dest_dir="/Users/mac/Downloads/sft/paper/sft_data"
num_workers=1

tokenizer_path="/mnt/public/open_source_AI/Meta-Llama-3.1-8B-Instruct"
python utils/tokenizer.py \
    --dataset_name ${dataset} \
    --dataset_path ${src_dir} \
    --output_path ${dest_dir} \
    --tokenizer_path ${tokenizer_path} \
    --version ${clearning_version} \
    --num_workers ${num_workers}
if [ $? -ne 0 ]; then
    echo "tokenizer.py failed."
    exit
else
    echo "tokenizer.py succeed."
fi

