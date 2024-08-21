import os
import json
import gzip
import argparse
import chardet
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from os import listdir, path
from utils.general_policy import GClean

_TEXT_LONG_REQUIRED_ = 10
cleaner = GClean(_TEXT_LONG_REQUIRED_)

def make_clean(args):
    global_file_no = 0
    global_id_no = 0

    jsonlfiles = sorted(listdir(args.source_path))
    for dir_no,subfile in tqdm(enumerate(jsonlfiles),total=len(jsonlfiles)):
       
        dest_file = os.path.join(args.dest_path,"part-39-{:06d}.jsonl".format(global_file_no))
        if os.path.exists(dest_file): os.remove(dest_file)
        global_file_no += 1
        of = open(dest_file,'w',encoding='utf-8')

        input_file = os.path.join(args.source_path,subfile)
        print("input_file:",input_file)
        with open(input_file, 'r',encoding='utf-8') as fin:
            for line in tqdm(fin):
                js_ = json.loads(line)
                '''
                {"question": "唐氏筛查afp值结果是0.81----（女24岁）", "answer": "你好，唐氏筛查如果mom值偏高的话，有可能胎儿不正常。建议您进一步做无创DNA的检查。这个是相对比较准确的。唐氏筛查跟很多因素有关系，比如您填写的数值身高体，体重，末次月经。大部分怀孕的胎儿是正常的。怀孕期间每一次的检查都是排除胎儿畸形的。"}
                '''
                js_dict = {}
                js_dict["id"] = global_id_no
                js_dict["source"] = "cn-medical-treatment"
                js_dict["subset"] = "39-health"
                js_dict["source_id"] = ""
                global_id_no += 1

                ques = js_["question"].strip()
                if ques[-1] not in ['。','！','？',"?","，",","]:
                    ques = ques + "？"
                else:
                    ques = ques[0:-1] + "？"
                answ = js_["answer"].strip()
                answ = cleaned_content = cleaner.clean_punct_at_begin(answ)
                js_dict["content"] = ques + answ

                print(json.dumps(js_dict,ensure_ascii=False),file=of)
                if of.tell() > args.max_size:
                    of.close()
                    dest_file = os.path.join(args.dest_path,"part-39-{:06d}.jsonl".format(global_file_no))
                    if os.path.exists(dest_file): os.remove(dest_file)
                    of = open(dest_file,'w',encoding='utf-8')
                    global_file_no += 1
    of.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path',
                        type=str,
                        default="/data/data_warehouse/SourceData/39_health",
                        help='Directory containing trained actor model')
    parser.add_argument('--dest_path',
                        type=str,
                        default="/localdisk/llm/source_data/cn-39-health",
                        help='Directory containing trained actor model')
    parser.add_argument('--dataset_name',
                        type=str,
                        default="cn-cn-39-health",
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

