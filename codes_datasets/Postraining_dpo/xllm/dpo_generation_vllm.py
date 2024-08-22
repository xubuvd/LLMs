# -*- coding:utf-8 -*-
import os
import glob
import re
import math
import json
import argparse
import random
from tqdm import tqdm
import hashlib
import time
import torch
from vllm import LLM, SamplingParams
import fastchat
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.conversation import Conversation, SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

os.environ["RAY_memory_monitor_refresh_ms"] = '0'

Model2PathDict = {
    "dpo_ckpt_70b_32k_v1":["/data/dpo_ckpt_70b_32k_v1","",""],
    "sft-70b_final_2311_97w_3epoch":["/data/sft70b_final_2311_97w_3epoch","",""],
    "dpo_ckpt_13b_bestChosenCurve":["/data/dpo_ckpt_13b_bestChosenCurve","",""],
    "dpo_ckpt_13b_32k_v5":["/data/dpo_ckpt_13b_32k_v5","",""],
    "dpo_ckpt_13b_32k_v4":["/data/dpo_ckpt_13b_32k_v4","",""],
    "dpo_ckpt_13b_32k_v6":["/data/dpo_ckpt_13b_32k_v6","",""],
    "dpo_ckpt_70b_32k_v6":["/data/dpo_ckpt_70b_32k_v6","",""],
    "dpo_ckpt_70b_32k_v8":["/data/dpo_ckpt_70b_32k_v8","",""],
    "dpo_ckpt_70b_32k_v7":["/data/dpo_ckpt_70b_32k_v7","",""],
    "dpo_ckpt_70b_32k_v9":["/data/dpo_ckpt_70b_32k_v9","",""],
    "dpo_ckpt_70b_32k_v10":["/data/dpo_ckpt_70b_32k_v10","",""],
    "sft-mistral7b-10epoch":["/data/sft-mistral7b-10epoch","",""],
    "dpo_ckpt_mistral-7b_32k_v1":["/data/dpo_ckpt_mistral-7b_32k_v1","",""],
    "dpo_ckpt_mistral-7b_32k_v4":["/data/dpo_ckpt_mistral-7b_32k_v4","",""],
    "dpo_ckpt_mistral-7b_32k_v5":["/data/dpo_ckpt_mistral-7b_32k_v5","",""],
    "dpo_ckpt_mistral-7b_32k_v6":["/data/dpo_ckpt_mistral-7b_32k_v6","",""],
    "dpo_ckpt_mistral-7b_32k_v7":["/data/dpo_ckpt_mistral-7b_32k_v7","",""],
    "dpo_ckpt_mistral-7b_32k_v8":["/data/dpo_ckpt_mistral-7b_32k_v8","",""],
    "dpo_ckpt_mistral-7b_32k_v9-500":["/data/dpo_ckpt_mistral-7b_32k_v9-500","",""],
    "dpo_ckpt_mistral-7b_32k_v9-1000":["/data/dpo_ckpt_mistral-7b_32k_v9-1000","",""],
    "dpo_ckpt_mistral-7b_32k_v9":["/data/dpo_ckpt_mistral-7b_32k_v9","",""],
    "dpo_ckpt_mistral-7b_32k_v10":["/data/dpo_ckpt_mistral-7b_32k_v10","",""],
    "dpo_ckpt_mistral-7b_32k_v12-1000":["/data/dpo_ckpt_mistral-7b_32k_v12-1000","",""],
    "dpo_ckpt_mistral-7b_32k_v12-2000":["/data/dpo_ckpt_mistral-7b_32k_v12-2000","",""],
    "dpo_ckpt_mistral-7b_32k_v12-3000":["/data/dpo_ckpt_mistral-7b_32k_v12-3000","",""],
    "dpo_ckpt_mistral-7b_32k_v12-4000":["/data/dpo_ckpt_mistral-7b_32k_v12-4000","",""],
    "dpo_ckpt_mistral-7b_32k_v12":["/data/dpo_ckpt_mistral-7b_32k_v12","",""],
    "dpo_ckpt_mistral-7b_sft5epoch_32k_v1":["/data/dpo_ckpt_mistral-7b_sft5epoch_32k_v1","",""],
    "dpo_ckpt_mistral-7b_sft5epoch_32k_v2":["/data/dpo_ckpt_mistral-7b_sft5epoch_32k_v2","",""],
    "product-sft-mistral7b-5epoch":["/data/sft-mistral7b-5epoch","",""],
    "product-dpo_ckpt_mistral-7b_sft5epoch_32k_v1-1k":["/data/dpo_ckpt_mistral-7b_sft5epoch_32k_v1-1k","",""],
    "dpo_ckpt_llama3-70b_sft3epoch_32k_v1":["/data/dpo_ckpt_llama3-70b_sft3epoch_32k_v1","",""],
    "dpo_ckpt_llama3-70b_sft3epoch_32k_v5":["/data/dpo_ckpt_llama3-70b_sft3epoch_32k_v5","",""],
    "product-dpo_ckpt_llama3-70b_sft3epoch_32k_v5_1k":["/data/dpo_ckpt_llama3-70b_sft3epoch_32k_v5_1k","",""],
    "dpo_ckpt_llama3-8b_5e6_v1":["/data/dpo_ckpt_llama3-8b_5e6_v1","",""],
    "dpo_ckpt_llama3-8b_5e7_v2":["/data/dpo_ckpt_llama3-8b_5e7_v2","",""],
    "dpo_ckpt_llama3-70b_5e6_3epoch":["/data/dpo_ckpt_llama3-70b_5e6_3epoch","",""],
    "dpo_ckpt_llama3-70b_5e6_6epoch":["/data/dpo_ckpt_llama3-70b_5e6_6epoch","",""]
}

PROMPT_DICT = {
    "prompt_input": (
        #"Below is an instruction that describes a task, paired with an input that provides further context. "
        #"Write a response that appropriately completes the request.\n\n"
        #"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        "<s>User: {instruction}\n{input}</s>\nAssistant:"
    ),
    "prompt_no_input": (
        #"Below is an instruction that describes a task. "
        #"Write a response that appropriately completes the request.\n\n"
        #"### Instruction:\n{instruction}\n\n### Response:"
        "<s>User: {instruction}</s>\nAssistant:"
    ),
}

def get_gen_prompt(model_path, datalist) -> str:
    conv = get_conversation_template(model_path)
    print("conv:",conv)
    '''
    conv: Conversation(
        name='mistral', 
        system_template='<s>{system_message}\n', 
        system_message='', 
        roles=('User', 'Assistant'), 
        messages=[], 
        offset=0, 
        sep_style=<SeparatorStyle.LLAMA2: 7>, 
        sep=' ', 
        sep2='</s>', 
        stop_str=None, 
        stop_token_ids=None)
    '''
    conv = Conversation(
        name=conv.name,
        system_template=conv.system_template,
        system_message=conv.system_message,
        roles=conv.roles,
        messages=list(conv.messages),  # prevent in-place modification
        offset=conv.offset,
        sep_style=SeparatorStyle(conv.sep_style),
        sep=conv.sep,
        sep2=conv.sep2,
        stop_str=conv.stop_str,
        stop_token_ids=conv.stop_token_ids,
    )

    for idx, message in enumerate(datalist):
        msg_role = "user" if idx % 2 == 0 else "assistant"
        if msg_role == "user":
            conv.append_message(conv.roles[0], message)
        elif msg_role == "assistant":
            conv.append_message(conv.roles[1], message)
        else:
            raise ValueError(f"Unknown role: {msg_role}")

    # Add a blank message for the assistant.
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    return prompt

def make_offline_models(args):
    model, sampling_params, model_path = init_model(args.model_name)
    output_list = []

    jsonl_list = []
    question_list = []
    with open(args.input_file, 'r',encoding='utf-8') as fin:
        for idx, line in enumerate(tqdm(fin)):
            line = line.strip()
            if len(line) < 10: continue
            
            js_dict = json.loads(line.strip())
            datalist = js_dict["data"]
            prompt = get_gen_prompt(model_path, datalist)

            print("idx:{}\tprompt:{}".format(idx,prompt))
            question_list.append(prompt)
            jsonl_list.append(js_dict)
    print(f"read {args.input_file} {len(jsonl_list)} lines.")

    if question_list:
        outputs = model.generate(question_list, sampling_params)
        for idx, output in tqdm(enumerate(outputs)):
            js_dict = jsonl_list[idx]
            answer = output.outputs[0].text.strip()
            js_dict['response'] = answer
            output_list.append(js_dict) 
    return output_list

def init_model(model_name):
    ngpus_per_node = torch.cuda.device_count()
    print("init model with {} gpus ...".format(ngpus_per_node))
    model_path,_,_ = Model2PathDict.get(model_name,None)
    model = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size = ngpus_per_node,
        gpu_memory_utilization = 0.9
    )
    print(f"model_path={model_path}")
    conv = get_conversation_template(model_path)
    print(f"conv={conv}")
    #tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.3, top_p=0.85, top_k=5, frequency_penalty=1.1, max_tokens=2048, stop=conv.stop_str, stop_token_ids=conv.stop_token_ids)
    return model, sampling_params, model_path#,tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_file',
        type=str,
        default="",
        required=True,
        help='dataset name'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default="frontis",
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default="",
        required=True,
        help='model'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help=""
    )
    parser.add_argument("--seed", type=int, default=3517, help="A seed for reproducible training.")
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    print("args:",args)

    output_list = make_offline_models(args)
    print(f"invoking model {args.model_name} ok, output_list: {len(output_list)}")

    model_layer = args.model_name
    output_dir =  os.path.join(f'./evaluation/results/{model_layer}', args.dataset_name)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    saved_name = "seed_" + str(args.seed) + ".json"
    output_file = os.path.join(output_dir, saved_name)
    if os.path.exists(output_file): os.remove(output_file)
    with open(output_file, "w", encoding='utf-8') as fo:
        json.dump(output_list, fo, indent=4, ensure_ascii=False)
 
