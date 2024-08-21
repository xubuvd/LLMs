# -*- encoding:utf-8 -*-
import os
import json
import multiprocessing as mp
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
import argparse
from os import listdir, path
from utils.general_policy import GClean
from utils.special_policy import SpecialPolicies
from tqdm import tqdm
from utils.util import load_set_from_txt
from emoji import emojize, demojize
from utils.check_black_words import CheckBlackWords
from utils.util import load_list_from_structedTxt, load_set_from_txt
from utils.clean_headtails_from_content import CleanHeadTailsFromContent

LONG_REQUIRED = 256 
_FLUSH_STEPS_NUM_ = 1000

cleaner = GClean(LONG_REQUIRED)
BadChecker = CheckBlackWords("/localdisk/datacleaner/utils/unikeyword.txt")

def step1(text):

    if len(text) < LONG_REQUIRED:
        return "step1", 0, text

    # text_normalization
    #cleaned_content = cleaner.text_normalization(text)

    # g1 InvalidWords + g22 ChineseLessThan60
    #cleaned_content = cleaner.clean_valid(cleaned_content)
    
    # deal with ![text](url)
    cleaned_content = cleaner.clean_special_url(text)

    # g0 CleanScript
    cleaned_content = cleaner.clean_script(cleaned_content)
    
    # g4 CleanDuplicatedPunctuation
    cleaned_content = cleaner.clean_deplicate_punc(cleaned_content)

    # g2 CleanPunctuationsAtHeadTail
    cleaned_content = cleaner.clean_punct_at_last(cleaned_content)
    cleaned_content = cleaner.clean_punct_at_begin(cleaned_content)

    # g3 EngPeriod2ChinPeriod
    #cleaned_content = re.sub('\\[.*?]', '。', cleaned_content)

    # g5 FanTi2Simplify
    #cleaned_content = re.sub('「', '“', cleaned_content)
    #cleaned_content = re.sub('」', '”', cleaned_content)
    #cleaned_content = re.sub('【', '[', cleaned_content)
    #cleaned_content = re.sub('】', ']', cleaned_content)

    # g16 CleanPersonInfoDoc
    cleaned_content = cleaner.clean_private(cleaned_content)

    # g6 CleanURL
    cleaned_content = cleaner.clean_url(cleaned_content)

    # g7 CleanContinueousPuncs
    cleaned_content = cleaner.clean_continueous_punc(cleaned_content)

    # g10 CleanDuplicationInText
    #cleaned_content = cleaner.delete_2repeating_long_patterns(cleaned_content)

    # g13 TooShortSentence
    cleaned_content = cleaner.filter_long_sentences(cleaned_content)
    
    #g20 TooLongSentence
    #has_punctuations = cleaner.is_chinese_long_strings_without_punctuation(cleaned_content)
    #if not has_punctuations:
    #    return "not_has_punctuations", 0, cleaned_content
    
    # g24 LongEnough
    if len(cleaned_content) >= LONG_REQUIRED:
        return "step1",1, cleaned_content
    return "step1", 0, cleaned_content

def step2(text):
    # # ["politician","badword","gumble","sex","ads","dirty"]
    is_sentive,key_words = BadChecker.is_spam_text(text,
        thresh_hold=3,
        black_dataType=["badword","gumble","sex","ads","dirty"]
    )
    if is_sentive:
        return "has_sensitive_words:{}".format(key_words),0,text

    #is_table_page = cleaner.is_TablePage(text)
    #if is_table_page:
    #    return "table_page",0,text

    return "",1,text

def step3(text):
    return "",1,text

def controller(args):
    
    check_file = os.path.join(args.dest_path,"done.log")
    if os.path.exists(check_file):
        done_set = load_set_from_txt(check_file)
    else:
        done_set = set()

    files = sorted(listdir(args.source_path))
    for line_no,input_file in tqdm(enumerate(files),total=len(files)):
        input_file = input_file.strip()
        if input_file in done_set: continue
        file_size = os.path.getsize(os.path.join(args.source_path,input_file))
        print(f"{input_file}-->{file_size}")
        if len(input_file) < 1 or file_size < 20: continue
        yield (args,input_file)

def make_clean(items):
    print(items)
    args,input_file = items
    
    clean_policy = ""
    clean_status = 0

    good_output_file = os.path.join(args.dest_path,"good",input_file)
    if os.path.exists(good_output_file): os.remove(good_output_file)
    good_fo = open(good_output_file, 'a+', encoding='utf-8')

    bad_output_file = os.path.join(args.dest_path,"bad",input_file)
    if os.path.exists(bad_output_file): os.remove(bad_output_file)
    bad_fo = open(bad_output_file, 'a+', encoding='utf-8')

    fpath = os.path.join(args.source_path,input_file)
    with open(fpath,"r",encoding="utf-8") as f:
        datas = f.readlines()
        for line_no,line in tqdm(enumerate(datas),total=len(datas)):
            line = line.strip()
            if len(line) < 1: continue
            
            js_dict = json.loads(line)

            content = js_dict["content"]
            clean_policy,clean_status,text = step1(content)
            #if clean_status == 1:
            #    clean_policy,clean_status,text = step2(text)

            js_dict["source"] = "en-wanjuan1.0"
            js_dict["source_id"] = ""
            js_dict["clean_policy"] = "{}".format(clean_policy if clean_status == 0 else "")
            js_dict["clean_status"] = clean_status
            js_dict["content"] = text
            jstr = json.dumps(js_dict, ensure_ascii=False)

            if clean_status == 1: good_fo.write(jstr+"\n")
            else: bad_fo.write(jstr+"\n")
            
            if line_no % _FLUSH_STEPS_NUM_ == 0:
                good_fo.flush()
                bad_fo.flush()
    good_fo.close()
    bad_fo.close()
    return input_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path',
                        type=str,
                        default="/data/data_warehouse/llm/llm-data-org.del/cn-wiki2_t2s",
                        help='Directory containing trained actor model')
    parser.add_argument('--dest_path',
                        type=str,
                        default="/data/data_warehouse/llm/clean_data/cn-wiki",
                        help='Directory containing trained actor model')
    parser.add_argument('--dataset_name',
                        type=str,
                        default="",
                        help="")
    parser.add_argument('--num_workers',
                        type=int,
                        default=32,
                        help="")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.dest_path):
        os.makedirs(args.dest_path, exist_ok=True)
    GoodDir = os.path.join(args.dest_path, "good")
    BadDir = os.path.join(args.dest_path, "bad")

    if not os.path.exists(GoodDir):
        os.makedirs(GoodDir, exist_ok=True)
    if not os.path.exists(BadDir):
        os.makedirs(BadDir,exist_ok=True)

    check_file = os.path.join(args.dest_path,"done.log")
    check_fo = open(check_file, 'a+', encoding='utf-8')

    pools = mp.Pool(args.num_workers)
    for res in pools.imap(make_clean, controller(args)):
        check_fo.write(res+"\n")
        check_fo.flush()
    pools.close()
    check_fo.close()

