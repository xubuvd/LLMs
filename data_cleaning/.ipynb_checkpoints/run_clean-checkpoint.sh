#!/bin/bash

dataset="cn-jd-product-rewrite"
clearning_version="v9"

source_dir="/localdisk/llm/source_data/${dataset}"
dest_dir="/localdisk/llm/clean_data/${dataset}/${clearning_version}"
num_workers=32

# Step1: Perform dataset cleaning
python clean/jdProduct_clean.py \
    --num_workers ${num_workers} \
    --dataset_name ${dataset} \
    --source_path ${source_dir} \
    --dest_path ${dest_dir}
if [ $? -ne 0 ]; then
    echo "${dataset}_clean.py failed."
    exit
else
    echo "${dataset}_clean.py succeed."
fi

# Step2: depupli amoung texts 
python text-dedup/text_dedup/minhash.py \
    --path ${source_dir} \
    --name ${dataset} \
    --output ${dest_dir} \
    --column content

# Step2: Make tokenizing with chinese_llama-1-13B, to yield ${dataset}-meta-info.json
tokenizer_path="/data/pangwei/chinese_llama_plus_13b_hf"
python utils/tokenizer.py \
    --dataset_name ${dataset} \
    --dataset_path ${dest_dir}/good \
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

# Step3: Sample 100 datas for evaluation, to produce ${dataset}-sample100.jsonl
python utils/random_sample.py \
    --dataset_name ${dataset} \
    --dataset_path ${dest_dir}/good \
    --output_path ${dest_dir} \
    --number_sample 100 \
    --version ${clearning_version}
if [ $? -ne 0 ]; then
    echo "random_sample.py failed."
    exit
else
    echo "random_sample.py succeed."
fi

