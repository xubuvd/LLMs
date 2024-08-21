import os
import json
import re
from tqdm import tqdm
import opencc
import argparse
from tqdm import tqdm
from os import listdir, path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path',
                        type=str,
                        default="/data/data_warehouse/llm/llm-data-org.del/cn-wiki2",
                        help='Directory containing trained actor model')
    parser.add_argument('--dest_path',
                        type=str,
                        default="/data/data_warehouse/llm/llm-data-org.del/",
                        help='Directory containing trained actor model')

    args = parser.parse_args()
    return args


def split_cn_wiki(args):
    files = sorted(listdir(args.source_path))

    WikiDir = os.path.join(args.dest_path, "cn-wiki2_t2s")
    if not os.path.exists(WikiDir):
        os.makedirs(WikiDir, exist_ok=True)

    converter = opencc.OpenCC('t2s.json')

    for input_file in tqdm(files,total=len(files)):

        ifile = os.path.join(args.source_path,input_file)

        wiki_output_file = os.path.join(WikiDir,input_file)
        if os.path.exists(wiki_output_file): os.remove(wiki_output_file)
        wiki_fo = open(wiki_output_file, 'a+', encoding='utf-8')

        for line in open(ifile,'r',encoding="utf-8"):
            line = line.strip()
            if len(line) < 5:continue
            js_dict = json.loads(line)
            content = converter.convert(js_dict["content"])
            js_dict["content"] = content
            jstr = json.dumps(js_dict, ensure_ascii=False)
            wiki_fo.write(jstr+"\n")
        wiki_fo.close()

if __name__ == '__main__':

    args = parse_args() 
    split_cn_wiki(args)

