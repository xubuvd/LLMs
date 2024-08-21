# -*- coding:utf-8 -*-
import os
import glob
import json
import argparse
from tqdm import tqdm
import multiprocessing as mp
import transformers

tokenizer_kwargs = {
    "use_fast": True,
    "revision": "xbGPT"
}
tokenizer_path="/mnt/public/open_source_AI/Meta-Llama-3.1-8B-Instruct"
tokenizer = transformers.AutoTokenizer.from_pretrained("/mnt/public/open_source_AI/Meta-Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

def jobj2count(jobj):
    """
        mp process controller
    """
    for itm in tqdm(jobj):
        yield itm

def process_file(js):
    global tokenizer
    num_tokens = 0
    text = ' '.join(js['data']).strip()
    tokens = tokenizer.encode(text,add_special_tokens=False)#(js['content'])
    num_tokens += len(tokens) 
    return {'num_tokens': num_tokens, "score": float(js['score'])}


def llama_tokenizer(args):
    input_dir = args.dataset_path
    src_files = sorted(glob.glob(os.path.join(input_dir, "*.jsonl"), recursive=True))
    print(f"src_files: {src_files}")

    pool = mp.Pool(args.num_workers)
    total_tokens = 0

    records = {}
    records["files"] = []

    for idx,xfile in tqdm(enumerate(src_files),total=len(src_files)):

        tokens = 0
        difficulty = 0.0
        filename = os.path.basename(xfile)#.replace(".jsonl","")
        print(f"process file: {filename}")
        
        with open(xfile,"r",encoding='utf-8') as fin:
            line_content = [json.loads(line) for line in fin.readlines()]
            for res in pool.imap(process_file, jobj2count(line_content)):
                tokens += res['num_tokens']
                difficulty += res['score']

            print(f'file {filename} has {tokens} tokens, {difficulty} difficulty scores.')
            records["files"].append(
                {
                    "filename":filename,
                    "llama_tokens":tokens,
                    "difficulty_scores":difficulty,
                    "total_samples":len(line_content),
                    "avg_tokens_per_sample":1.0*tokens/len(line_content),
                    "avg_difficulty_score_per_sample":1.0*difficulty/len(line_content),
                }
            )
        total_tokens += tokens
    records["total_llama_tokens"] = total_tokens
    return records

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name',
                        type=str,
                        default="jdItem",
                        help='dataset name')
    parser.add_argument('--dataset_path',
                        type=str,
                        default="/data_warehouse/llm/source_data/JDItem_pattern_dataset/SampledRawDataset/",
                        help='source path')
    parser.add_argument('--output_path',
                        type=str,
                        default="/data_warehouse/llm/source_data/JDItem_pattern_dataset/",
                        help='source path')

    parser.add_argument('--tokenizer_path',
                        type=str,
                        default="/xxxx/chinese_llama_13b_plus84",
                        help="tokenizer path, default LLaMA tokenizer")
    parser.add_argument('--version',
                        type=str,
                        default="v1",
                        help=""
    )
    parser.add_argument('--num_workers',
                        type=int,
                        default=32,
                        help="")
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    tokenizer_kwargs = {
        "use_fast": True,
        "revision": "productGPT"
    }

    args = parse_args()
    records = {}

    #tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path, **tokenizer_kwargs)
    #tokenizer.pad_token = tokenizer.eos_token
    print(f"num of llama tokens: {tokenizer.vocab_size}")

    records = llama_tokenizer(args)
    records['dataset'] = args.dataset_name

    output_file = os.path.join(args.output_path,"{}-meta-info-{}.json".format(args.dataset_name,args.version))
    if os.path.exists(output_file): os.remove(output_file)
    with open(output_file, 'w') as f:
        json.dump(records, f, indent=4)

