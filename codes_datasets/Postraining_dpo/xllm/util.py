import os
import glob
import json
import random
import torch
import numpy as np
import transformers
from tqdm import tqdm
from typing import Dict, Optional
from datasets import Dataset
from transformers import set_seed
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    tokenizer_path: Optional[str] = field(default="facebook/opt-125m")
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )

@dataclass
class DataArguments:
    train_dataset_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    test_dataset_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    sanity_check: Optional[bool] = field(default=True, metadata={"help": "only train on 1000 samples"})
    
    data_suffix : str = field(
        default=None, metadata={"help": "Path to the trained file to be loaded"}
    )
    
    model_architecture_type: str = field(
        default=None, metadata={"help": "required token: LLama3, LLama2, Qwen"}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    
    #train_dataset_path: Optional[str] = field(default="", metadata={"help": "dataset path for training, expected formation: jsonl"})
    #test_dataset_path:  Optional[str] = field(default="", metadata={"help": "dataset path for evaluation, expected formation: jsonl"})

    output_dir: Optional[str] = field(default="", metadata={"help": "model saved path"})
    optimizer: Optional[str] = field(default="adamw_hf", metadata={"help": "optimizer used in training"})

    # debug argument for distributed training
    remove_unused_columns: bool = field(default=False)
    logging_first_step: bool = field(default=True)
    bf16: bool = field(default=True)
    lr_scheduler_type: str = field(default="cosine")
    max_grad_norm: float = field(default=1.0)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default= 0.95)
    weight_decay: float = field(default=0.1)
    gradient_checkpointing_kwargs = {'use_reentrant':False} # OR gradient_checkpointing_kwargs={'use_reentrant':True}

@dataclass
class xllmArguments:
    enable_flash_attn: bool = field(default=True)
    tie_word_embeddings: bool = field(default=False)

    only_train_embedding: Optional[bool] = field(default=False)
    only_debug_dataload: Optional[bool] = field(default=False)
    
    max_length: Optional[int] = field(default=2048, metadata={"help": "max length of each sample"})
    max_prompt_length: Optional[int] = field(default=1024, metadata={"help": "max length of each sample's prompt"})
    max_target_length: Optional[int] = field(
        default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})


def extract_prompt(prompt_and_response):
    search_term = "<|im_start|>assistant\n"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    
    return prompt_and_response[: search_term_idx + len(search_term)]

def read_jsonl_lst(input_file_lst,chunksize=64*1024):
    jsonl_list = []
    for indx,input_file in tqdm(enumerate(input_file_lst),total=len(input_file_lst)):
        objects_list = []
        with open(input_file, 'r',encoding='utf-8') as fin:
            for idx, line in enumerate(tqdm(fin)):
                line = line.strip()
                if len(line) < 100: continue
                js_dict = json.loads(line)
                objects_list.append(js_dict)
        print("loading {} json objects from {}".format(len(objects_list),input_file))
        jsonl_list.extend(objects_list)
    return jsonl_list

def get_dataset(
        input_file: str, 
        split: str, 
        sanity_check: bool = False, 
        silent: bool = False, 
        cache_dir: str = None,
        data_suffix: str = None,
        model_type: str = None,
        max_sample: str = None) -> Dataset:

    jsonl_list = []
    if os.path.isdir(input_file):
        input_file_lst = sorted(glob.glob(os.path.join(input_file,data_suffix), recursive=True))
    elif os.path.isfile(input_file):
        input_file_lst = [input_file]
    else:
        print(f"Invalid-format: {input_file}")
        return None
    jsonl_list = read_jsonl_lst(input_file_lst)
    if sanity_check: jsonl_list = jsonl_list[0:min(len(jsonl_list), 1000)]

    if max_sample is None: max_sample = len(jsonl_list)
    random.seed(3517)
    jsonl_list = list(random.sample(jsonl_list, max_sample))

    dpo_dataset_dict = {
        "prompt":[],
        "chosen":[],
        "rejected":[],
    }
    for idx, js_dict in tqdm(enumerate(jsonl_list),total=len(jsonl_list)):
        prompt = js_dict["prompt"]
        chosen = js_dict["chosen"]
        rejected = js_dict["rejected"]

        dpo_dataset_dict["prompt"].append(prompt)
        dpo_dataset_dict["chosen"].append(chosen)
        dpo_dataset_dict["rejected"].append(rejected)

    assert len(dpo_dataset_dict["prompt"]) == len(dpo_dataset_dict["chosen"])
    assert len(dpo_dataset_dict["chosen"]) == len(dpo_dataset_dict["rejected"])

    dpo_data = Dataset.from_dict(dpo_dataset_dict)
    return dpo_data

# Random seeds for reproducability
def set_random_seed(seed=None):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

