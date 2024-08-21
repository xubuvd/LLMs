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

def random_sample_benchmark(sample_num,input_dir):
    
    files = sorted(glob.glob(os.path.join(input_dir,"v*"), recursive=True))
    good_dir = os.path.join(input_dir,files[-1],"good")

    input_files = sorted(glob.glob(os.path.join(good_dir,"*.jsonl"), recursive=True))
    avg_nums_per_task = math.ceil(sample_num/len(input_files))

    sample_1000 = []
    for file in tqdm(input_files,total=len(input_files)):
        filename = os.path.basename(file).replace(".jsonl","")
        data = []
        for line in open(file,"r",encoding='utf-8'):
            try:
                js = json.loads(line.strip())
            except json.decoder.JSONDecodeError:
                print(line)
            data.append(js)
        random.shuffle(data)
        print(f"process file {filename}, total of {len(data)}.")
        sample_1000.extend(data[0:avg_nums_per_task])       
    return sample_1000

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name',
                        type=str,
                        default="SafetyCheck",
                        help='dataset name')
    parser.add_argument('--dataset_path',
                        type=str,
                        default="/data/data_warehouse/llm/clean_data/",
                        help='source path')
    parser.add_argument('--output_path',
                        type=str,
                        default="./",
                        help='source path')

    parser.add_argument("--number_sample",
        type=int,
        default=100000,
        help="number of sampled data"
    )
    parser.add_argument('--version',
                        type=str,
                        default="v1",
                        help=""
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    sample_benchmark = []

    subdirs = sorted(glob.glob(os.path.join(args.dataset_path,"cn-*"), recursive=True))
    print(f"{subdirs}")
    for subdir in subdirs:
        dir_ = os.path.join(args.dataset_path,subdir)
        print(f"Processing {dir_}...")
        samples = random_sample_benchmark(
            sample_num=args.number_sample,
            input_dir=dir_
        )
        sample_benchmark.extend(samples)
    print(f"sampling benchmark questions: {len(sample_benchmark)}")

    output_file=f"{args.output_path}/{args.dataset_name}-sample-2k.jsonl"
    if os.path.exists(output_file): os.remove(output_file)
    fo = open(output_file, 'w', encoding='utf-8')

    random.shuffle(sample_benchmark)
    samples = sample_benchmark[0:2000]

    for idx, item in tqdm(enumerate(samples),total=len(samples)):
        item["id"] = idx + 1
        jstr = json.dumps(item, ensure_ascii=False)
        fo.write(jstr+"\n")
    fo.close()
    print(f"Output file {output_file}, total sampled {len(samples)}")

