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
       
        dest_file = os.path.join(args.dest_path,"part-{:06d}.jsonl".format(global_file_no))
        if os.path.exists(dest_file): os.remove(dest_file)
        global_file_no += 1
        of = open(dest_file,'w',encoding='utf-8')

        input_file = os.path.join(args.source_path,subfile)
        print("input_file:",input_file)
        with open(input_file, 'r',encoding='utf-8') as fin:
            for line in tqdm(fin):
                js_ = json.loads(line)
                '''
                {"question": "康宝xdr53-tvc1消毒柜使用方法", "answers": "、使用前认真检查设备运转是否正常，调节器和显示器是否“失控”。2、把洗净、抹净余水的餐具、茶具、食具按平行排列方式倒放或斜放于柜内架层上。3、关好柜门接通电源，扭动起动键。4、扭动“起动”键后，石英管开始发亮，表示消毒工作开始，消 毒结束后，自动切断电源，15分钟后才能打开门取用餐具。", "category": "生活"}
                {"question": "临期的香水可以买吗", "answers": "最好不要买吧，因为香水这种东西还挺耐用的，不可能快速就用完，有可能过期了也只用了一点点，小毫升的可以买，因为很快消耗掉，所以没关系，特别大的一瓶就没必要买了，买香水最好提前试香，选最喜欢的买，避开不喜欢的味道，没必要追求便宜去买的", "category": "生活"}
                '''
                js_dict = {}
                js_dict["id"] = global_id_no
                js_dict["source"] = "cn-sina-iask"
                js_dict["subset"] = js_["category"].strip()
                js_dict["source_id"] = ""
                global_id_no += 1

                ques = js_["question"].strip()
                if ques[-1] not in ['。','！','？',"?","，",","]:
                    ques = ques + "？"
                else:
                    ques = ques[0:-1] + "？"
                answ = js_["answers"].strip()
                answ = cleaned_content = cleaner.clean_punct_at_begin(answ)
                js_dict["content"] = ques + answ

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
                        default="/data/data_warehouse/SourceData/sina_iask",
                        help='Directory containing trained actor model')
    parser.add_argument('--dest_path',
                        type=str,
                        default="/localdisk/llm/source_data/cn-sina-iask",
                        help='Directory containing trained actor model')
    parser.add_argument('--dataset_name',
                        type=str,
                        default="cn-sina-iask",
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

