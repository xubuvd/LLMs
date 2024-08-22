#!bin/bash


models=("dpo_ckpt_llama-70b_5e6_3epoch" "dpo_ckpt_llama-70b_5e6_6epoch")
for model in ${models[*]}
do
    echo "infer model ${model} on test dataset of ${dataset_name} ..."
    CUDA_VISIBLE_DEVICES='0,1' python dpo_generation_vllm.py \
        --input_file ../dpo_dataGen/xllm_eval_500_final.jsonl \
        --model_name $model \
        --batch_size 4
done

