import os
import json
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

    subfiles = sorted(listdir(args.source_path))
    for dir_no,subfile in tqdm(enumerate(subfiles),total=len(subfiles)):
        
        input_file = os.path.join(args.source_path,subfile)

        with open(input_file, 'r') as f:
            datalist = f.readlines()

        for line in datalist:
            line = line.strip()
            if len(line) < 1:
                continue

            js_data = json.loads(line)
            js_dict = {}
            js_dict["id"] = global_id_no
            js_dict["source"] = "cn-baidu-weixin"
            js_dict["source_id"] = js_data['url']
            js_dict["subset"] = js_data["search_keyword"]
            js_dict["content"] = js_data["content"]
            global_id_no += 1

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
                        default="/data/data_warehouse/SourceData/baidu_weixin/231027/",
                        help='Directory containing trained actor model')
    parser.add_argument('--dest_path',
                        type=str,
                        default="/localdisk/llm/source_data/cn-baidu-weixin",
                        help='Directory containing trained actor model')
    parser.add_argument('--dataset_name',
                        type=str,
                        default="cn-baidu-weixin",
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

