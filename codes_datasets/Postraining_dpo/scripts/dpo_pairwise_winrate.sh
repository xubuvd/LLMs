#!/bin/bash

models=("dpo_ckpt_llama-70b_5e6_3epoch" "dpo_ckpt_llama-70b_5e6_6epoch")
for model_compared in ${models[*]}
do
    for eval_file in 'xllmtest'
    do
        k1=sft-checkpoint-72800
        k2=$model_compared

        python win_tie_loss_stat.py \
            -i1 ${k1}-${k2}-${eval_file}.json \
            -k1 $k1 \
            -i2 ${k2}-${k1}-${eval_file}.json \
            -k2 $k2 \
            --output_dir ./ \
            --dst $eval_file
    done
done

