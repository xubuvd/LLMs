# coding=utf-8
# Copyright 2023 The XLLM Inc. team. All rights reserved.

import os
import sys
import socket
import torch
import pathlib
import transformers
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from trl import DPOTrainer

from util import (
    get_dataset,
    ModelArguments,
    DataArguments,
    TrainingArguments,
    xllmArguments,
    set_random_seed
)

def train():
 
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, xllmArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        xllm_args,
    ) = parser.parse_args_into_dataclasses()
    logger.info(xllm_args)

    #if data_args.model_architecture_type not in ["Qwen", "Llama"]:
    #    print(f"Don't support model {data_args.model_architecture_type} training now")
    #    return
    # load the DPOTrainer expected dataset
    train_dataset = get_dataset(
        input_file=data_args.train_dataset_path,
        split="train",
        sanity_check=data_args.sanity_check,
        data_suffix=data_args.data_suffix,
        model_type=data_args.model_architecture_type)
    print(f" size of train_dataset:{len(train_dataset)}")
    print("train_dataset:{}".format(train_dataset["prompt"][0:2]))

    # load evaluation dataset
    eval_dataset = get_dataset(
        input_file=data_args.test_dataset_path,
        split="test",
        sanity_check=data_args.sanity_check,
        data_suffix=data_args.data_suffix,
        model_type=data_args.model_architecture_type)
    print(f"size of eval_dataset:{len(eval_dataset)}")

    # load a pretrained model trust_remote_code=True
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        #use_flash_attention_2=xllm_args.enable_flash_attn,
        use_safetensors=True
    ).to('cuda')
    model.train()
    
    if model_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # Load instruction tuned model
    model_ref = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        use_safetensors=True,
    ).to('cuda')

    # Load a tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    
    # DPO training used for Llama and Qwen, except for Mistral-7b
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        #tokenizer.pad_token_id = 0

    model.config.use_cache = False
    model.config.tie_word_embeddings = xllm_args.tie_word_embeddings
    logger.warning(f"{model.config.tie_word_embeddings=}")

    logger.warning(
        f"Resize token embedding from {model.vocab_size} to {tokenizer.vocab_size}"
    )
    # model.resize_token_embeddings(tokenizer.vocab_size) #fixed for Assertion `srcIndex < srcSelectDimSize` failed.

    # initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=xllm_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=xllm_args.max_length,
        max_target_length=xllm_args.max_target_length,
        max_prompt_length=xllm_args.max_prompt_length,
        generate_during_eval=True
    )

    # Post-Training with Direct Preference Optimization
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        dpo_trainer.train(resume_from_checkpoint=True)
    else:
        dpo_trainer.train()
    dpo_trainer.save_state()
    dpo_trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    global_rank = os.environ["RANK"]
    ip = socket.gethostbyname(socket.gethostname())
    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=True)
    logger.add(
        f"./logs/pretrain.{ip}.rank_{global_rank}.log", level="DEBUG", colorize=True
    )
    set_random_seed(seed=3517)

    logger.info("Start DPO training")
    with logger.catch():
        train()

