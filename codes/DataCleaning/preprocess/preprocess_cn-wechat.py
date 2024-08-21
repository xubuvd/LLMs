import re
import os
import json
#import jieba_fast as jieba
#import gzip
import argparse
#import chardet
from tqdm import tqdm
from os import listdir, path

def get_head_tail_sentence(args):
    global_file_no = 0
    global_id_no = 0

    dest_file = os.path.join(args.dest_path,"wechat_content_sentences.txt")
    if os.path.exists(dest_file): os.remove(dest_file)
    of = open(dest_file,'w',encoding='utf-8')

    subsets = sorted(listdir(args.source_path))
    for dir_no,file_name in tqdm(enumerate(subsets),total=len(subsets)):
       
        input_file = os.path.join(args.source_path,file_name)
        with open(input_file, 'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if len(line) < 100:continue
                js_dict = json.loads(line)
                content = js_dict["content"].strip()
                if len(content) < 100: continue
                
                '''
                split_flg = [',',';','。',',','；','。','！','？',' ','\n','\t']
                
                fpos = 1
                while fpos < len(content) and content[fpos] not in split_flg: fpos += 1
                head = content[0:fpos]
                
                lpos = len(content) - 1 -1
                while lpos > 0 and content[lpos] not in split_flg: lpos -= 1
                tail = content[lpos+1:]
                '''
                head = content[50:len(content)-50]
                #if len(head) > args.topk: head = head[:args.topk]
                #if len(tail) > args.topk: tail = tail[-args.topk:]
                print(head,file=of)
                #if tail != head: print(tail,file=of)
    of.close()

def text_segment(args):
    # /root/llm/source_data/wechat_head_tail_sentences.txt
    dest_file = os.path.join("/root/llm/source_data/","wechat_head_tail_sentences_segment.txt")
    if os.path.exists(dest_file): os.remove(dest_file)
    of = open(dest_file,'w',encoding='utf-8')

    with open("/root/llm/source_data/wechat_head_tail_sentences.txt", 'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) < 3:continue
            seg_list = jieba.cut(line,cut_all=False)
            text = ' '.join([item for item in seg_list if len(item) > 1])
            print(text,file=of)
    of.close()

def extract_keyphrase(args):
    keyphrse_dict = dict()

    idx = 0
    with open("/root/llm/source_data/phrases.txt",'r') as f:
        for line in tqdm(f):
            line = line.strip()
            if len(line) < 1:continue
            tokens = line.split("\t")
            if len(tokens) != 3: 
                print("tokens:",tokens)
                continue
            phrase = tokens[1].replace("_","")
            if phrase not in keyphrse_dict:
                keyphrse_dict[phrase] = [1,tokens[2]]
            else:
                keyphrse_dict[phrase][0] = keyphrse_dict[phrase][0] + 1
            idx += 1
            #if idx > 50000: break
    #
    keyphrse_list = sorted(keyphrse_dict.items(), key = lambda kv:(kv[1], kv[0]),reverse = True)
    for item in keyphrse_list:
        # ('眼下正是', [1, '102.464'])
        freq = item[1][0]
        muinfo = item[1][1]
        phrase = item[0]
        #if freq < 100: continue
        print(f"{phrase}\t{freq}\t{muinfo}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path',
                        type=str,
                        default="/data/data_warehouse/llm/source_data/cn-wechat",
                        help='Directory containing trained actor model')
    parser.add_argument('--dest_path',
                        type=str,
                        default="/root/llm/source_data/",
                        help='Directory containing trained actor model')
    parser.add_argument('--dataset_name',
                        type=str,
                        default="cn-wechat",
                        help="")
    parser.add_argument('--topk',
                        type=int,
                        default=20,
                        help="max chunk size")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.dest_path):
        os.makedirs(args.dest_path, exist_ok=True)
    #get_head_tail_sentence(args)
    #text_segment(args)
    extract_keyphrase(args)

