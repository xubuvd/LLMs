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
    
    files = sorted(glob.glob(os.path.join(input_dir,"*.jsonl"), recursive=True))
    
    avg_nums_per_task = math.ceil(sample_num/len(files))

    sample_1000 = []
    for file in tqdm(files,total=len(files)):
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
                        default="xhs",
                        help='dataset name')
    parser.add_argument('--dataset_path',
                        type=str,
                        default="/yuan1.0/open_source_1T",
                        help='source path')
    parser.add_argument('--output_path',
                        type=str,
                        default="./",
                        help='source path')

    parser.add_argument("--number_sample",
        type=int,
        default=100,
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

    sample_benchmark = random_sample_benchmark(
        sample_num=args.number_sample,
        input_dir=args.dataset_path
    )
    print(f"sampling benchmark questions: {len(sample_benchmark)}")

    output_file=f"{args.output_path}/{args.dataset_name}-sample{args.number_sample}-{args.version}.jsonl"
    if os.path.exists(output_file): os.remove(output_file)
    fo = open(output_file, 'w', encoding='utf-8')

    for idx, item in tqdm(enumerate(sample_benchmark),total=len(sample_benchmark)):
        item["id"] = idx + 1
        jstr = json.dumps(item, ensure_ascii=False)
        fo.write(jstr+"\n")
    fo.close()
    print(f"Output file {output_file}, total sampled {len(sample_benchmark)}")

