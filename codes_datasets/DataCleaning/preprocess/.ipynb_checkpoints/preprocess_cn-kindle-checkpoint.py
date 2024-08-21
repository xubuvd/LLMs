import os
import json
import argparse
import chardet
from tqdm import tqdm
from os import listdir, path



def make_clean(args):
    global_file_no = 0
    global_id_no = 0

    subsets = sorted(listdir(args.source_path))
    for dir_no,subset_dir in tqdm(enumerate(subsets),total=len(subsets)):
        
        #subset_dir = subset_dir.replace(" ","\ ")
        file_dir = os.path.join(args.source_path,subset_dir)

        dest_file = os.path.join(args.dest_path,"part-{:06d}.jsonl".format(global_file_no))
        if os.path.exists(dest_file): os.remove(dest_file)
        global_file_no += 1
        of = open(dest_file,'w',encoding='utf-8')
        
        for root, dirs, files in os.walk(file_dir):
            print('root_dir:', root)
            print('files:', files)
       
            #root = root.replace(" ","\ ")
            for file in files:
                #file = file.replace(" ","\ ")
                if not (file.endswith(".txt") or file.endswith(".shtml")): continue
                input_file = os.path.join(root,file)

                html_str = open(input_file, 'rb').read()
                encoding_info = chardet.detect(html_str)
                original_encoding = encoding_info['encoding']
                if original_encoding not in ["UTF-8","GB2312","GB18030","Big5","utf-8","UTF-16","UTF-32"]: continue

                html_str = html_str.decode(original_encoding, 'ignore')#.encode('utf-8')
                if len(html_str) < 512: continue

                js_dict = {}
                js_dict["id"] = global_id_no
                js_dict["source"] = "cn-kindle"
                js_dict["subset"] = subset_dir
                js_dict["source_id"] = input_file
                global_id_no += 1

                js_dict["content"] = html_str

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
                        default="/data/data_warehouse/llm/source_data/cn-kindle",
                        help='Directory containing trained actor model')
    parser.add_argument('--dest_path',
                        type=str,
                        default="/data/data_warehouse/llm/source_data/cn-kindle2",
                        help='Directory containing trained actor model')
    parser.add_argument('--dataset_name',
                        type=str,
                        default="cn-kindle",
                        help="")
    parser.add_argument('--max_size',
                        type=int,
                        default=500 * 1024 * 1024,
                        help="max chunk size")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.dest_path):
        os.makedirs(args.dest_path, exist_ok=True)
    make_clean(args)



