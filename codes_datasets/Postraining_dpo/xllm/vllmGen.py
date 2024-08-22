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
import asyncio
import torch
import openai
from vllm import LLM, SamplingParams
import fastchat
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.conversation import Conversation, SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
import request_http 
import data_decrypt

from init_models import (
     InitModel_AquilaChat2,
     InitModel_Baichuan2,GetCompletion_Baichuan2,
     InitModel_Qwen,GetCompletion_Qwen,
     InitModel_chatglm,GetCompletion_chatglm,
     InitModel_Llama2,GetCompletion_Llama2,GetCompletion_Llama3,
     InitModel_internlm,GetCompletion_internlm,
     InitModel_BELLE,GetCompletion_BELLE,
     InitModel_Yi,GetCompletion_Yi
)
from async_openai_resquest import (
    package_messages,
    batch_openai,
    MultiThreadAsyncOpenAI
)
from util import read_jsonl_md5

os.environ["RAY_memory_monitor_refresh_ms"] = '0'

MAX_API_RETRY=10

Model2PathDict = {
    "Qwen-14B-Chat":["/mnt/public/open_source_AI/Qwen-14B-Chat",InitModel_Qwen,GetCompletion_Qwen],
    "Qwen-7B-Chat":["/mnt/public/open_source_AI/Qwen-7B-Chat",InitModel_Qwen,GetCompletion_Qwen],
    "Qwen1.5-72B-Chat":["/mnt/public/open_source_AI/Qwen1.5-72B-Chat",InitModel_Qwen,GetCompletion_Qwen],
    "Qwen-1_8B-Chat":["/mnt/public/open_source_AI/Qwen-1_8B-Chat",InitModel_Qwen,GetCompletion_Qwen],
    "Qwen1.5-32B-Chat":["/mnt/public/open_source_AI/Qwen1.5-32B-Chat",InitModel_Qwen,GetCompletion_Qwen],
    "Qwen2-72B-Instruct":["/mnt/public/open_source_AI/Qwen2-72B-Instruct",InitModel_Qwen,GetCompletion_Qwen],
    "Qwen1.5-14B-Chat":["/mnt/public/open_source_AI/Qwen1.5-14B-Chat",InitModel_Qwen,GetCompletion_Qwen],
    "sftExp1-Qwen-14B":["/mnt/models/sftExp1-Qwen-14B/checkpoint-1335",InitModel_Qwen,GetCompletion_Qwen],
    "sftExp2-Qwen-14B":["/mnt/models/sftExp2-Qwen-14B/checkpoint-final",InitModel_Qwen,GetCompletion_Qwen],
    "sftExp3-Qwen-14B":["/mnt/models/sftExp3-Qwen-14B/checkpoint-2575",InitModel_Qwen,GetCompletion_Qwen],
    "sftExp4-Qwen-14B":["/mnt/models/sftExp4-Qwen-14B/checkpoint-final",InitModel_Qwen,GetCompletion_Qwen],
    "sftExp5-Qwen-14B":["/mnt/models/sftExp5-Qwen-14B/checkpoint-final",InitModel_Qwen,GetCompletion_Qwen],
    "sftExp6-Qwen-14B":["/mnt/models/sftExp6-Qwen-14B/checkpoint-final",InitModel_Qwen,GetCompletion_Qwen],
    "sftExp7-Qwen-14B":["/mnt/models/sftExp7-Qwen-14B/sftExp7-Qwen-14B-checkpoint-2052",InitModel_Qwen,GetCompletion_Qwen],
    "sftExp8-Qwen-14B":["/mnt/models/sftExp8-Qwen-14B/sftExp8-Qwen-14B-checkpoint-3712",InitModel_Qwen,GetCompletion_Qwen],
    "sftExp8.1-Qwen1.5-14B-Chat":["/mnt/models/sftExp8.1-Qwen1.5-14B-Chat/sftExp8.1-Qwen1.5-14B-Chat-checkpoint-final-4epoch",InitModel_Qwen,GetCompletion_Qwen],
    "sftExp8.2-Qwen1.5-14B-Chat":["/mnt/models/sftExp8.2-Qwen1.5-14B-Chat/sftExp8.2-Qwen1.5-14B-Chat-checkpoint-final",InitModel_Qwen,GetCompletion_Qwen],
    "sftExp8.3-Qwen1.5-14B":["/mnt/models/sftExp8.3-Qwen1.5-14B/sftExp8.3-Qwen1.5-14B-checkpoint-final",InitModel_Qwen,GetCompletion_Qwen],
    "sftExp8.3-Qwen1.5-14B-Chat":["/mnt/models/sftExp8.3-Qwen1.5-14B-Chat/sftExp8.3-Qwen1.5-14B-Chat-checkpoint-final",InitModel_Qwen,GetCompletion_Qwen],
    "Qwen2-7B-Instruct":["/mnt/public/open_source_AI/Qwen2-7B-Instruct",InitModel_Qwen,GetCompletion_Qwen],
    "qwen1.5-14b":["/mnt/public/open_source_AI/qwen1.5-14b",InitModel_Qwen,GetCompletion_Qwen],
     "Qwen2-1.5B-Instruct":["/mnt/public/open_source_AI/Qwen2-1.5B-Instruct",InitModel_Qwen,GetCompletion_Qwen],
    "chatglm3-6b":["/mnt/public/open_source_AI/chatglm3-6b",InitModel_chatglm,GetCompletion_chatglm],
    "chatglm_pro":["/mnt/public/open_source_AI/chatglm_pro",InitModel_chatglm,GetCompletion_chatglm],
    "Baichuan-13B-Chat":["/mnt/public/open_source_AI/Baichuan-13B-Chat",InitModel_Baichuan2,GetCompletion_Baichuan2],
    "Baichuan2-13B-Chat":["/mnt/public/open_source_AI/Baichuan2-13B-Chat",InitModel_Baichuan2,GetCompletion_Baichuan2],
    "Yi-1.5-9B-Chat":["/mnt/public/open_source_AI/Yi-1.5-9B-Chat",InitModel_Yi,GetCompletion_Yi],
    "Meta-Llama-3.1-70B-Instruct":["/mnt/public/open_source_AI/Meta-Llama-3.1-70B-Instruct",InitModel_Llama2,GetCompletion_Llama3],
    "Meta-Llama-3.1-8B-Instruct":["/mnt/public/open_source_AI/Meta-Llama-3.1-8B-Instruct",InitModel_Llama2,GetCompletion_Llama3],
    "Llama-2-7b-chat-hf":["/mnt/public/open_source_AI/Llama-2-7b-chat-hf",InitModel_Llama2,GetCompletion_Llama2],
    "Meta-Llama-3.1-405B-Instruct":["/mnt/public/open_source_AI/Meta-Llama-3.1-405B-Instruct",InitModel_Llama2,GetCompletion_Llama3],
    "Meta-Llama-3-70B-Instruct":["/mnt/public/open_source_AI/Meta-Llama-3-70B-Instruct",InitModel_Llama2,GetCompletion_Llama3],
    "Meta-Llama-3-8B-Instruct":["/mnt/public/open_source_AI/Meta-Llama-3-8B-Instruct",InitModel_Llama2,GetCompletion_Llama3]
}

def get_gen_prompt(model_path, datalist) -> str:
    conv = get_conversation_template(model_path)
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

def request_chatgpt(model_name,messages):
    wait_base = 0.1
    content = "Exception"
    for _ in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(
                #选择模型
                model=model_name,
                messages=messages
            )
            content = response["choices"][0]["message"]["content"]
        except Exception as e:
            print("Exception:{}".format(e),flush=True)
            time.sleep(wait_base)
            wait_base = wait_base*1.2
        else:
            break
    return content


def load_single_file(data_file, pd_key,chunksize=64 * 1024):

    bchunk_per_file = data_decrypt.get_binary_content_from_file(data_file, pd_key, chunksize)  ## 来自 data_decrypt.py     
    schunk_per_file = bchunk_per_file.decode("utf-8", "ignore")     
    print(f"extracting json string from {data_file} ...")     
    print('schunk_per_file:', len(schunk_per_file))     
    json_list = data_decrypt.extract_json_objects(schunk_per_file)     
    return json_list

def decrypt_all_files(args, pd_key):     
    objects_list = []      
    src_files = sorted(glob.glob(os.path.join(args.input_dir, "*.enc"), recursive=True))     
    print(f"src_files:{src_files}")     
    
    for idx, xfile in tqdm(enumerate(src_files), total=len(src_files)):         
        json_list = load_single_file(xfile, pd_key)     
        objects_list.extend(json_list)     
    return objects_list

def load_done_dataset(args):
    done_id_set = set()
    
    output_dir = os.path.join(args.input_dir,args.output_path)
    check_file = os.path.join(output_dir,args.tmp_file_suffix)
    
    if os.path.exists(output_dir) and os.path.exists(check_file):
        print(f"check_file:{check_file}")
        with open(check_file, 'r', encoding='utf-8') as fin:
            for idx, line in enumerate(tqdm(fin)):
                line = line.strip()
                if len(line) < 10: continue
                try:
                    js_dict = json.loads(line)
                except json.decoder.JSONDecodeError: continue
                pid = str(js_dict['id']).strip() + "|#|" + js_dict['source'].strip()
                done_id_set.add(pid)
    return done_id_set

def make_offline_models(args,jsonl_list):
    
    model, sampling_params, model_path, tokenizer = init_model(args.model_name,args)
    
    data_len = len(jsonl_list)
    print(f'jsonl_list{data_len}')
    
    output_list = []
    prompt_list = []
    done_id_set = load_done_dataset(args)
    
    for idx, js_dict in tqdm(enumerate(jsonl_list),total=len(jsonl_list)):
        pid = str(js_dict['id']).strip() + "|#|" + js_dict['source'].strip()
        if pid in done_id_set:
            print(f"pid is done: {pid}")
            output_list.append(js_dict)
            continue
        if args.model_name not in js_dict["selected_tdhc_models"]:
            output_list.append(js_dict)
            continue
        if args.model_name in js_dict and len(js_dict[args.model_name].strip()) > 0:
            output_list.append(js_dict)
            continue
        prompt_list.append(js_dict)
    print(f"split finished for model: {args.model_name} #######")
    
    question_list = []
    new_prompt_list = []
    for idx,js_dict in tqdm(enumerate(prompt_list),total=len(prompt_list)):
        datalist = js_dict["data"]
        prompt = get_gen_prompt(model_path, datalist)
        #tokens_num = len(tokenizer.encode(prompt))
        #if tokens_num <= 2000:
        question_list.append(prompt)
        new_prompt_list.append(js_dict)
    print(f"prompt generation finished for model {args.model_name} #######")
    print(f"question_list: {len(question_list)}")
    
    if question_list:
        # question_list=question_list[0:2]
        outputs = model.generate(question_list, sampling_params)
        assert len(outputs) == len(new_prompt_list), "outputs: {} is not equal to new_prompt_list:{}".format(len(outputs),len(new_prompt_list))

        # print(outputs[0].prompt, outputs[0].outputs[0].text)
        for idx, output in tqdm(enumerate(outputs),total=len(outputs)):
            js_dict = new_prompt_list[idx]
            answer = output.outputs[0].text
            js_dict[args.model_name] = answer.strip()
            output_list.append(js_dict)
    return output_list

def make_openai(args,fo):

    jsonl_list = []
    with open(args.input_dir, 'r',encoding='utf-8') as fin:
        for idx, line in enumerate(tqdm(fin)):
            line = line.strip()
            if len(line) < 10: continue
            js_dict = json.loads(line.strip())
            jsonl_list.append(js_dict)
    print(f"read {args.input_dir} {len(jsonl_list)} lines.")

    not_model_list = []
    message_list = []
    for idx, js_dict in tqdm(enumerate(jsonl_list),total=len(jsonl_list)):

        if args.model_name not in js_dict["selected_models"]:
            not_model_list.append(js_dict)
            continue
        if args.model_name in js_dict and js_dict[args.model_name][0] != "Exception" and len(js_dict[args.model_name]) > 0:
            not_model_list.append(js_dict)
            continue
        message_list.append(js_dict)

    for idx, js_dict in tqdm(enumerate(not_model_list),total=len(not_model_list)):
        print(json.dumps(js_dict,ensure_ascii=False),file=fo)
    fo.flush()

    output_list = []
    for idx, js_dict in tqdm(enumerate(message_list),total=len(message_list)):
        data_list = js_dict['data']
        message = []
        for idx in range(0,len(data_list)):
            if idx % 2 == 0:
                message.append({"role": "user", "content": data_list[idx]})
            elif idx % 2 == 1:
                message.append({"role": "assistant", "content": data_list[idx]})
        
        res = request_chatgpt(args.model_name,message)
        print("{}: message:{}, res:{}".format(args.model_name, message, res))
        js_dict[args.model_name] = res
        print(json.dumps(js_dict,ensure_ascii=False),file=fo)
        if idx % 10 == 0: fo.flush()
        del js_dict['data']
        output_list.append(js_dict)
    return output_list

def init_model(model_name,args):
    print("init model ...")
    ngpus_per_node = torch.cuda.device_count()
    model_path,init_model,_ = Model2PathDict.get(model_name,None)
    model = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size = ngpus_per_node,
        dtype='bfloat16',
        #quantization='gptq',
        gpu_memory_utilization = 0.9
    )
    conv = get_conversation_template(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=0.1, 
        top_p=0.85, 
        top_k=20, 
        frequency_penalty=1.1, 
        max_tokens=args.max_tokens, 
        stop=conv.stop_str, 
        stop_token_ids=conv.stop_token_ids)
    return model, sampling_params, model_path,tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir',
        type=str,
        default="",
        required=True,
        help='dataset name'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default="",
        required=False,
        help='dataset name'
    )
    parser.add_argument(
        '--models_name_list',
        nargs='+',
#         type=list,
#         nargs='+',
        required=True,
        help='model'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=5,
        help=""
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help=""
    )
    parser.add_argument(
        '--tmp_file_suffix',
        type=str,
        default="tmp.jsonl",
        required=False,
        help=""
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default="0.1"
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=8192,
    )
    parser.add_argument(
        '--presence_penalty',
        type=float,
        default="1.1"
    )
    parser.add_argument(
        '--frequency_penalty',
        type=float,
        default="1.2"
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default="1.0"
    )
    parser.add_argument(
        '--tok',
        default="47e32629d38293d2"
    )
    parser.add_argument(
         '--done_file_name',
         default="dpo_check.txt"
     ) 
    parser.add_argument(
         '--output_path',
         default=""
     )
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    
    args = parse_args()
    print("args.models_name_list:",args.models_name_list)
    
    pd_key_dict = request_http.get_pd_token(args.tok)  ## get_pd_token 来自 request_http.py      
    pd_key = pd_key_dict['data'].encode('utf-8')      
    jsonl_list = decrypt_all_files(args,pd_key)

    for idx, model_name in tqdm(enumerate(args.models_name_list),total=len(args.models_name_list)):
        print(f"processing {model_name} ...")
        args.model_name = model_name.strip()
        if len(args.model_name) < 2: continue
        args.tmp_file_suffix = "{}_tmp.jsonl".format(args.model_name)
        
        if args.model_name in ["gpt-3.5-turbo-0301", "gpt-3.5-turbo", "gpt-3.5-turbo-0613", "gpt-4-1106-preview"]:
            messages_list,origin_message_list = package_messages(args)
            print("the Num. of messages: {}".format(len(messages_list)))
            assert len(messages_list) == len(origin_message_list)
            MultiThreadAsyncOpenAI(args,messages_list,origin_message_list)
        else:
            output_list = make_offline_models(args,jsonl_list)
            print(f"invoking model {args.model_name} ok, output_list: {len(output_list)}")

            output_dir = os.path.join(args.input_dir.replace(".jsonl",""),args.output_path)
            if not os.path.exists(output_dir): os.mkdir(output_dir)
            output_file = os.path.join(output_dir,args.tmp_file_suffix)

            print(f'output_file:{output_file}')
            if os.path.exists(output_file): os.remove(output_file)
            fo = open(output_file, 'w', encoding='utf-8')

            for idx, item in tqdm(enumerate(output_list),total=len(output_list)):
                del item['data']
                print(json.dumps(item,ensure_ascii=False),file=fo)
                if idx % 10 == 0: fo.flush()
            fo.close()

