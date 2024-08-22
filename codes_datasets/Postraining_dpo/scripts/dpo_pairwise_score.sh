#!/bin/bash

# gpt-3.5-turbo-0613
# gpt-4-0613

models=("dpo_ckpt_llama-70b_5e6_3epoch" "dpo_ckpt_llama-70b_5e6_6epoch")
for model in ${models[*]}
do
    for eval_file in 'frontis'
    do
        k1=sft-70b_final_95w_3epoch
        k2=$model
        scorer=gpt-4-0613

        echo "pairwise compare between ${model} and ${k1} on ${eval_file} ..."
        python xllm/dpo_pairwise_score_by_gpt4.py \
            -i1 ./evaluation/results/${k1}/${eval_file}/seed_3517.json \
            -i2 ./evaluation/results/${k2}/${eval_file}/seed_3517.json \
            -k1 $k1 \
            -k2 $k2 \
            --batch_size 10 \
            --max_tokens 32 \
            --output_dir ./ \
            --eval_scorer $scorer
        
        python xllm/dpo_pairwise_score_by_gpt4.py \
            -i1 ./evaluation/results/${k2}/${eval_file}/seed_3517.json \
            -i2 ./evaluation/results/${k1}/${eval_file}/seed_3517.json \
            -k1 $k2 \
            -k2 $k1 \
            --batch_size 10 \
            --max_tokens 32 \
            --output_dir ./ \
            --eval_scorer $scorer
    done
done

