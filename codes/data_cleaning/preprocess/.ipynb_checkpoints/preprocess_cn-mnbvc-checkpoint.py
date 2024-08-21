import os
import json
import gzip
import argparse
import chardet
from tqdm import tqdm
from os import listdir, path

def make_clean(args):
    global_file_no = 0
    global_id_no = 0

    subsets = sorted(listdir(args.source_path))
    for dir_no,subset_dir in tqdm(enumerate(subsets),total=len(subsets)):
       
        if subset_dir not in ["gov","law","news","qa"]: continue

        file_dir = os.path.join(args.source_path,subset_dir)

        dest_file = os.path.join(args.dest_path,"part-{:06d}.jsonl".format(global_file_no))
        if os.path.exists(dest_file): os.remove(dest_file)
        global_file_no += 1
        of = open(dest_file,'w',encoding='utf-8')
        
        for root, dirs, files in os.walk(file_dir):
            print('root_dir:', root)
            print('files:', files)
            for file in files:
                if not file.endswith(".jsonl.gz"):continue
                input_file = os.path.join(root,file)
                print("input_file:",input_file)
                with gzip.open(input_file, 'rt') as f:
                    for line in f:
                        js_ = json.loads(line)
                        
                        js_dict = {}
                        js_dict["id"] = global_id_no
                        js_dict["source"] = "cn-mnbvc"
                        js_dict["subset"] = subset_dir
                        js_dict["source_id"] = file
                        global_id_no += 1
                        
                        if subset_dir in ["gov"]:
                            if "文件名" in js_:
                                js_dict["source_id"] = js_["文件名"]
                                js_dict["content"] = '\n'.join([item["内容"] for item in js_["段落"]])
                            else:
                                js_dict["source_id"] = eval(js_["meta"])["文件名"]
                                js_dict["content"] = js_["text"]
                        elif subset_dir in ["law"]:
                            js_dict["source_id"] = js_["分卷名"]
                            js_dict["content"] = js_["详情"]
                        elif subset_dir in ["news"]:
                            js_dict["source_id"] = os.path.basename(js_["文件名"])
                            js_dict["content"] = '\n'.join([item["内容"] for item in js_["段落"]])
                        elif subset_dir in ["qa"]:
                            js_dict["source_id"] = js_["来源"]
                            js_dict["content"] = js_["问"]+"\n"+js_["答"]

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
                        default="/data/data_warehouse/llm/source_data/cn-mnbvc",
                        help='Directory containing trained actor model')
    parser.add_argument('--dest_path',
                        type=str,
                        default="/data/data_warehouse/llm/source_data/cn-mnbvc2",
                        help='Directory containing trained actor model')
    parser.add_argument('--dataset_name',
                        type=str,
                        default="cn-mnbvc",
                        help="")
    parser.add_argument('--max_size',
                        type=int,
                        default=200 * 1024 * 1024,
                        help="max chunk size")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.dest_path):
        os.makedirs(args.dest_path, exist_ok=True)
    make_clean(args)

