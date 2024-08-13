import os
import json
import multiprocessing as mp
# import re
from tqdm import tqdm
import argparse
from os import listdir, path
import sys
sys.path.append(r"..")
from utils.general_policy import GClean
from utils.special_policy import SpecialPolicies
# from read_xhs_note_data import read_data, read_excel
from tqdm import tqdm
# from textrank4zh import TextRank4Keyword, TextRank4Sentence
import regex as re
# from summerizer import truncate_sentence, remove_initial_symbols
# from jieba.analyse import extract_tags
from emoji import emojize, demojize
from utils.check_black_words import SENSITIVE_WORDS

# tr4s = TextRank4Sentence()

LONG_REQUIRED = 256

sensitive_words = SENSITIVE_WORDS

cleaner = GClean(LONG_REQUIRED)

def val_pro(val):
    if val == 'null' or val is None:
        return 0
    else:
        return int(val)


def jobj2clean(jobj):
    """
        mp process controller
    """
    for itm in tqdm(jobj):
        yield itm


def controller(input_file):
    print(input_file)
    with open(os.path.join(args.source_path, input_file),"r",encoding="utf-8") as f:
        datalist = f.readlines()
        # print(datalist)
        for line in tqdm(datalist):
            # print(line)
            if len(line.strip())<1:
                continue
            try:
                line = json.loads(line)
            except Exception as e:
                print(e)
                print(line)            
            yield (line,input_file)

def step1(text):

    if len(text) < LONG_REQUIRED:
        return "step1", 0, text
    # Special strategy
    # cleaned_content = zdmcleaner.delete_date(text)
    cleaned_content = SpecialPolicies.delete_like_collect_comment(text)
    cleaned_content = SpecialPolicies.delete_author_claim(cleaned_content)
    if not SpecialPolicies.detect_lottery(cleaned_content):
        return "detect_lottery", 0, cleaned_content
    # g1 InvalidWords 
    cleaned_content = cleaner.clean_valid(cleaned_content)
    cleaned_content = emojize(cleaned_content)

    # g4 CleanDuplicatedPunctuation
    cleaned_content = cleaner.clean_deplicate_punc(cleaned_content)

    
    # print(cleaned_content)
    # g2 CleanPunctuationsAtHeadTail
    cleaned_content = cleaner.clean_punct_at_last(cleaned_content)
    cleaned_content = cleaner.clean_punct_at_begin(cleaned_content)

    # g3 EngPeriod2ChinPeriod
    # cleaned_content = re.sub('\\[.*?]', '。', cleaned_content)

    # g5 FanTi2Simplify
    cleaned_content = re.sub('「', '“', cleaned_content)
    cleaned_content = re.sub('」', '”', cleaned_content)
    cleaned_content = re.sub('【', '[', cleaned_content)
    cleaned_content = re.sub('】', ']', cleaned_content)

    # g16 CleanPersonInfoDoc
    cleaned_content = cleaner.clean_private(cleaned_content)

    # g6 CleanURL
    cleaned_content = cleaner.clean_url(cleaned_content)

    # g7 CleanContinueousPuncs
    cleaned_content = cleaner.clean_continueous_punc(cleaned_content)
    # print(cleaned_content)
    # g10 CleanDuplicationInText
    cleaned_content = cleaner.delete_2repeating_long_patterns(cleaned_content)
    cleaned_content = cleaner.deleteSpaceBetweenChinese(cleaned_content)
    # print(cleaned_content)
    # g13 TooShortSentence
    cleaned_content = cleaner.filter_long_sentences(cleaned_content)
    # print(cleaned_content)
    # g20 TooLongSentence
    cleaned_content = cleaner.remove_long_strings_without_punctuation(cleaned_content)

    # g24 LongEnough
    if len(cleaned_content) < LONG_REQUIRED:
        return "step1", 0, cleaned_content

    # g22 ChineseLessThan60
    if cleaner.ChineseLessThan60(cleaned_content):
        cleaned_content = cleaner.ChineseLessThan60(cleaned_content)
        return "step1", 1, cleaned_content
    else:
        return "ChineseLessThan60", 0, cleaned_content
    

def step2(text):
    cleaned = cleaner.remove_strings_with_keywords(text, sensitive_words)
    if cleaned:
        return "step2", 1, text
    else:
        return "remove_with_keywords", 0, text

def step3():
    pass
def make_clean(items):
    # print('make cleaning')
    line,input_file = items
    # print(type(line), line)
    try:
        text = demojize(line['content'])
    except Exception as e:
        print('demojize error')
        print(e)
        print(line)
    clean_policy = ""
    clean_status = 0

    # step 1: text normalization
    clean_policy,clean_status,text = step1(text)
    if clean_status == 1:
        # step 2: low quality
        clean_policy,clean_status,text = step2(text)
    #     if clean_status == 1:
    #         # step 3: text duplication
    #         clean_policy,clean_status,text = step3(text)


    return {
        "id":line['id'],
        "source_id":line['source_id'],
        "source":line['source'],
        "subset":line['subset'],
        "clean_policy":"{}".format(clean_policy if clean_status == 0 else ""),
        "clean_status":clean_status,
        "content":text,
    }

def HandleSingleFile(input_file, good_fo, bad_fo):
    pools = mp.Pool(PROCESS)

    flush_steps = 0
    flush_per_steps = 50
    print(input_file)
    for res in pools.imap(make_clean, controller(input_file)):
        jstr = json.dumps(res, ensure_ascii=False)
        if res["clean_status"] == 1:
            good_fo.write(jstr+"\n")
        else:
            bad_fo.write(jstr+"\n")
        flush_steps += 1
        if flush_steps % flush_per_steps == 0:
            good_fo.flush()
            bad_fo.flush()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path',
                        type=str,
                        default="/data1/zhouchuan/zdm",
                        help='Directory containing trained actor model')
    parser.add_argument('--dest_path',
                        type=str,
                        default="/data1/zhouchuan/zdm_V1",
                        help='Directory containing trained actor model')
    parser.add_argument('--num_workers',
                        type=int,
                        default=64,
                        help='# cpus')
    parser.add_argument('--dataset_name',
                        type=str,
                        default="zdm",
                        help='dataset name')    

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    PROCESS = args.num_workers
    files = sorted(listdir(args.source_path))

    GoodDir = os.path.join(args.dest_path, "good")
    BadDir = os.path.join(args.dest_path, "bad")

    if not os.path.exists(GoodDir):
        os.makedirs(GoodDir, exist_ok=True)
    if not os.path.exists(BadDir):
        os.makedirs(BadDir,exist_ok=True)

    for input_file in tqdm(files,total=len(files)):
        if 'nfs' in input_file:
            continue
        # input_file = os.path.join(args.source_path, input_file)
        good_output_file = os.path.join(GoodDir, input_file)
        if os.path.exists(good_output_file): os.remove(good_output_file)
        good_fo = open(good_output_file, 'a+', encoding='utf-8')

        bad_output_file = os.path.join(BadDir, input_file)
        if os.path.exists(bad_output_file): os.remove(bad_output_file)
        bad_fo = open(bad_output_file, 'a+', encoding='utf-8')

        HandleSingleFile(input_file, good_fo, bad_fo)

        good_fo.close()
        bad_fo.close()
