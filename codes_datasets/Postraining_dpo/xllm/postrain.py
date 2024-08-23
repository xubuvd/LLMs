# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import socket
import logging
import multiprocessing
from loguru import logger
from contextlib import nullcontext

from trl.commands.cli_utils import DPOScriptArguments, init_zero_verbose, TrlParser
from trl.env_utils import strtobool

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import PartialState
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from util import (
    get_dataset,
    set_random_seed,
    DataArguments
)
from request_http import get_pd_token

def train():
    parser = TrlParser((DPOScriptArguments, DPOConfig, ModelConfig,DataArguments))
    args, training_args, model_config, data_args = parser.parse_args_and_config()

    logger.info("args: {}".format(args))
    logger.info("training_args: {}".format(training_args))
    logger.info("model_config: {}".format(model_config))
    logger.info("data_args: {}".format(data_args))

    if data_args.model_architecture_type not in ["Qwen", "Llama"]:
        print(f"Don't support model {data_args.model_architecture_type} training now")
        return
    # pull pd_token from LP Com.
    pd_key_dict = get_pd_token(data_args.pd_token)
    pd_key = pd_key_dict['data'].encode('utf-8') 
    
    # load the DPOTrainer expected dataset
    train_dataset = get_dataset(
        input_file=data_args.train_dataset_path,
        split="train",
        sanity_check=args.sanity_check,
        data_suffix=data_args.data_suffix,
        pd_key=pd_key,
        model_type=data_args.model_architecture_type)
    print(f"size of train_dataset:{len(train_dataset)}")
    print("train_dataset:{}".format(train_dataset["prompt"][0:2]))

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        **model_kwargs
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        **model_kwargs
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    #if tokenizer.chat_template is None:
    #    tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"
    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,#eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

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

