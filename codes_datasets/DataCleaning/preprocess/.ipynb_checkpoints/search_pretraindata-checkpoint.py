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


    dest_file = os.path.join(args.dest_path,"part-{:06d}.jsonl".format(global_file_no))
    if os.path.exists(dest_file): os.remove(dest_file)
    global_file_no += 1
    of = open(dest_file,'w',encoding='utf-8')

    subsets = sorted(listdir(args.source_path))
    for dir_no,subset_dir in tqdm(enumerate(subsets),total=len(subsets)):

        if subset_dir.find("cn-") == -1: continue

        file_dir = os.path.join(args.source_path,subset_dir)
        for root, dirs, files in os.walk(file_dir):
            print('root_dir:', root)
            print('files:', files)
            for file in files:
                if not file.endswith(".jsonl"):continue
                input_file = os.path.join(root,file)
                print("input_file:",input_file)
                with open(input_file, 'r',encoding='utf-8') as f:
                    for line in f:
                        js_dict = json.loads(line)
                       
                        content = js_dict["content"]
                        if content.find("时代在召唤") == -1 or content.find("长城Assistant") == -1: continue

                        if content.find("时代在召唤") >= 0:
                            js_dict["datatype"] = "时代在召唤"
                        elif content.find("长城Assistant") >= 0:
                            js_dict["datatype"] = "长城Assistant"

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
                        default="/llm-data-org.del/",
                        help='Directory containing trained actor model')
    parser.add_argument('--dest_path',
                        type=str,
                        default="/localdisk/datacleaner/preprocess/",
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

