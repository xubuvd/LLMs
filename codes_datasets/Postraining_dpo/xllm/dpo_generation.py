from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig, set_seed
import torch
import argparse
import json
import os
from tqdm import tqdm
import random
import numpy as np
from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model

PROMPT_DICT = {
    "prompt_input": (
        #"Below is an instruction that describes a task, paired with an input that provides further context. "
        #"Write a response that appropriately completes the request.\n\n"
        #"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        "<|im_start|>system\n你是由XLLM开发的人工智能助手。<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    ),
    "prompt_no_input": (
        #"Below is an instruction that describes a task. "
        #"Write a response that appropriately completes the request.\n\n"
        #"### Instruction:\n{instruction}\n\n### Response:"
        #"<s>User: {instruction}</s>\nAssistant:"
        "<|im_start|>system\n你是由XLLM开发的人工智能助手。<|im_end|>\n<|im_start|>user\n{instruction}<|im    _end|>\n<|im_start|>assistant\n"
    ),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help="",
        required=False,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--seed", type=int, default=3517, help="A seed for reproducible training.")
    args = parser.parse_args()
    return args

# Random seeds for reproducability
def set_random_seed(seed=None):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main(args=None):
    
    set_random_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    no_split_module_classes = LlamaForCausalLM._no_split_modules
    config = LlamaConfig.from_pretrained(args.model_name_or_path)
    with init_empty_weights():
        model = LlamaForCausalLM._from_config(config, torch_dtype=torch.bfloat16) #加载到meta设备中，不需要耗时，不需要消耗内存和显存
    
    device_map = infer_auto_device_map(model,no_split_module_classes=no_split_module_classes) #自动划分每个层的设备
    load_checkpoint_in_model(model,args.model_name_or_path,device_map=device_map) #加载权重
    model = dispatch_model(model,device_map=device_map) #并分配到具体的设备上

    tokenizer = LlamaTokenizer.from_pretrained(
        args.tokenizer_path,
        use_fast=True,
        #trust_remote_code=True
    )
    torch.set_grad_enabled(False)
    model.eval()
    
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    print(prompt_input) 
    if args.dataset_name == "vicuna":
        dataset_path = '../dpo_dataGen/vicuna_test_set.jsonl'
        prompt_key = 'text'
    elif args.dataset_name == "koala":
        dataset_path = '../dpo_dataGen/koala_test_set.jsonl'
        prompt_key = 'prompt'
    elif args.dataset_name == "sinstruct":
        dataset_path = '../dpo_dataGen/sinstruct_test_set.jsonl'
        prompt_key = 'instruction'
    elif args.dataset_name == "wizardlm":
        dataset_path = '../dpo_dataGen/wizardlm_test_set.jsonl'
        prompt_key = 'Instruction'
    elif args.dataset_name == "truthfulqa":
        dataset_path = '../dpo_dataGen/truthfulqa_test_set.jsonl'
        prompt_key = 'Question'
    elif args.dataset_name == "anthropic":
        dataset_path = '../dpo_dataGen/anthropic_test_set.jsonl'
        prompt_key = 'instruction'
    elif args.dataset_name == "oasst":
        dataset_path = '../dpo_dataGen/oasst_test_set.jsonl'
        prompt_key = 'instruction'
    elif args.dataset_name == "frontis":
        dataset_path = "../dpo_dataGen/frontis_eval_500_final.jsonl"
        prompt_key = "data"

    with open(dataset_path,"r",encoding="utf-8") as fo:
        results = []
        dataset = list(fo)
        for point in tqdm(dataset,total=len(dataset)):
            point = json.loads(point)
            instruction = point[prompt_key]
            print(instruction) 
            if args.dataset_name == "sinstruct":
                instances = point['instances']
                assert len(instances) == 1
                if  instances[0]['input']:
                    prompt = prompt_input.format_map({"instruction":instruction, 'input':instances[0]['input']})
                else:
                    prompt = prompt_no_input.format_map({"instruction":instruction})
            elif args.dataset_name in ['anthropic', 'oasst']:
                prompt = prompt_input.format_map({"instruction":instruction, 'input':point['input']})
            elif args.dataset_name == "frontis":
                prompt = prompt_no_input.format_map({"instruction":instruction[0]})
            else:
                prompt = prompt_no_input.format_map({"instruction":instruction})
            
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(device)
            
            print("generating...")
            generate_ids = model.generate(
                input_ids,
                max_length=1024
            )
            print("decoding...")
            outputs = tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0]
            point['raw_output'] = outputs.strip()
            point['response'] = outputs.split("Assistant:")[1].strip()#outputs.split("Response:")[1]
            results.append(point)
            print("response: {}".format(point['response']))
            print("-"*60,flush=True)
    
    model_layer = args.model_name_or_path.split('/')[-1]
    output_dir =  os.path.join(f'./evaluation/results/{model_layer}', args.dataset_name)
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    saved_name = "seed_" + str(args.seed) + ".json"
    output_file = os.path.join(output_dir, saved_name)
    if os.path.exists(output_file): os.remove(output_file)
    with open(output_file, "w", encoding='utf-8') as fo:
        json.dump(results, fo, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)

