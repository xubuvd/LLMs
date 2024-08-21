import os
import json
import gzip
import argparse
import chardet
from tqdm import tqdm
from os import listdir, path

def split(args):
    global_file_no = 0
    global_id_no = 0

    dest_file = os.path.join(args.dest_path,"part-{:06d}.jsonl".format(global_file_no))
    if os.path.exists(dest_file): os.remove(dest_file)
    of = open(dest_file,'w',encoding='utf-8')

    subsets = sorted(listdir(args.source_path))
    for dir_no,file_name in tqdm(enumerate(subsets),total=len(subsets)):
       
        input_file = os.path.join(args.source_path,file_name)
        with open(input_file, 'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if len(line) < 1:continue
                js_dict = json.loads(line)
                #js_dict["id"] = js_dict["note_id"]
                #del js_dict["note_id"] 
                print(json.dumps(js_dict,ensure_ascii=False),file=of)
                if of.tell() > args.max_size:
                    of.close()
                    dest_file = os.path.join(args.dest_path,"part-{:06d}.jsonl".format(global_file_no))
                    if os.path.exists(dest_file): os.remove(dest_file)
                    of = open(dest_file,'w',encoding='utf-8')
                    global_file_no += 1
    of.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path',
                        type=str,
                        default="/data/tianqingxiang/data/llm/ocr/ocr_infer_result/200W",
                        help='Directory containing trained actor model')
    parser.add_argument('--dest_path',
                        type=str,
                        default="/root/llm/source_data/cn-JD-ocrtext",
                        help='Directory containing trained actor model')
    parser.add_argument('--dataset_name',
                        type=str,
                        default="cn-JD-ocrtext",
                        help="")
    parser.add_argument('--max_size',
                        type=int,
                        default=200*1024*1024,
                        help="max chunk size")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.dest_path):
        os.makedirs(args.dest_path, exist_ok=True)
    split(args)

