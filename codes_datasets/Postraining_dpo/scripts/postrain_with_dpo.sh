#!/bin/bash
set -e

log_out=0
only_print=0
dist_only_print=0
enable_flash_attn="True"
tie_embed="False"

datestr=`date +"%Y-%m-%d"`
wandb_run_name="dpo-sftExp8.3-Qwen1.5-14B-cp1006-$datestr"

output_path=/mnt/sftExp8.3-Qwen1.5-14B-checkpoint-1006-post
ckpt_path=/mnt/sftExp8.3-Qwen1.5-14B/sftExp8.3-Qwen1.5-14B-checkpoint-1006
model_type="Qwen"

data_suffix="*.jsonl"
train_data_path=/mnt/xubu/dpo_dataGen/dpo_preference_data/train
dev_data_path=/mnt/xubu/dpo_dataGen/dpo_preference_data/dev

num_processes=32
beta=0.1
bs_per_dev=2
grad_acc_steps=2

# save model per 500 global_step (2B token, 3h)
ckpt_steps=283
eval_steps=283

# Direct Preference Optimization
train_epoch=3
lr=5e-6
warmup_ratio=0.02

max_length=2048
max_prompt_length=1024
max_target_length=1024

strategy=zero3
sanity_check=False

# Run Command
REPO=$(pwd)
config=$REPO/scripts/accelerate_configs/${strategy}_multi_nodes.yaml
echo "config file: $config"

CMD=""

CMD="$CMD PYTHONPATH=$REPO"
CMD="$CMD accelerate launch"

CMD="$CMD --num_processes=$num_processes --config_file=$config"
CMD="$CMD $REPO/xllm/postrain.py"

CMD="$CMD --beta $beta --model_name_or_path $ckpt_path --learning_rate $lr --model_architecture_type $model_type"
CMD="$CMD --per_device_train_batch_size $bs_per_dev --gradient_accumulation_steps $grad_acc_steps"

CMD="$CMD --max_length $max_length --max_prompt_length $max_prompt_length --max_target_length $max_target_length"
CMD="$CMD --sanity_check $sanity_check --report_to 'wandb' --run_name $wandb_run_name"

CMD="$CMD --ignore_bias_buffers False --logging_steps 1 --tie_word_embeddings $tie_embed --enable_flash_attn $enable_flash_attn"

CMD="$CMD --train_dataset_path $train_data_path --test_dataset_path $dev_data_path --data_suffix $data_suffix"

CMD="$CMD --eval_steps $eval_steps --num_train_epochs $train_epoch --warmup_ratio $warmup_ratio"

CMD="$CMD --output_dir $output_path --save_steps $ckpt_steps"

CMD="$CMD --gradient_checkpointing True --weight_decay 0.1 --max_grad_norm 1.0"


echo $CMD
printf "===== Running Command =====\n"
printf "\t%s\n\n" "$CMD"

if [[ $only_print == "0" ]]; then
    printf "===== Command Logs =====\n"
    if [[ $log_out == "1" ]]; then
        echo "Command is running...."
        echo "Please run [tail -f ${log_file}] in another shell to monitoring the running process."
    fi
    if [[ -d $REPO/logs ]]; then
        timestamp=$(date +"%Y%m%d.%H.%M.%S")
        mv $REPO/logs $REPO/logs.$timestamp
    fi
    mkdir $REPO/logs
    eval "$CMD"
fi

