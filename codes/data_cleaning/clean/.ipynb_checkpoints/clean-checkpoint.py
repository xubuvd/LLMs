import sys
sys.path.append(r"..")
import os
import json
import multiprocessing as mp
# import re
from tqdm import tqdm
import argparse
from os import listdir, path
from utils.general_clean import GClean
from utils.special_policy import SpecialPolicies
from utils.read_xhs_note_data import read_data, read_excel
from tqdm import tqdm
# from textrank4zh import TextRank4Keyword, TextRank4Sentence
import regex as re
# from summerizer import truncate_sentence, remove_initial_symbols
# from jieba.analyse import extract_tags
from emoji import emojize, demojize
import multiprocessing as mp
from sensitive_words import SENSITIVE_WORDS

#
_TEXT_LONG_REQUIRED_ = 100

#
_PROCESS_ = 64

#
_FLUSH_STEPS_NUM_ = 1000

sensitive_words = SENSITIVE_WORDS

cleaner = GClean(_TEXT_LONG_REQUIRED_)

def step1(text):

    if len(text) < 50:
        return "step1", 0, text
    # Special strategy
    # cleaned_content = zdmcleaner.delete_date(text)
    cleaned_content = zdmcleaner.delete_like_collect_comment(cleaned_content)
    cleaned_content = zdmcleaner.delete_author_claim(cleaned_content)

    if len(cleaned_content) < _TEXT_LONG_REQUIRED_:
        return "TooShortSentence", 0, text

    # g0 CleanScript
    cleaned_content = cleaner.clean_script(text)
    
    # g4 CleanDuplicatedPunctuation, not used for cn-wiki dataset
    # cleaned_content = cleaner.clean_deplicate_punc(cleaned_content)

    # g1 InvalidWords + g22 ChineseLessThan60
    cleaned_content = cleaner.clean_valid(cleaned_content)

    # g2 CleanPunctuationsAtHeadTail
    cleaned_content = cleaner.clean_punct_at_last(cleaned_content)
    cleaned_content = cleaner.clean_punct_at_begin(cleaned_content)

    # g3 EngPeriod2ChinPeriod
    cleaned_content = re.sub('\\[.*?]', '。', cleaned_content)

    # g5 FanTi2Simplify
    cleaned_content = re.sub('「', '“', cleaned_content)
    cleaned_content = re.sub('」', '”', cleaned_content)
    cleaned_content = re.sub('【', '[', cleaned_content)
    cleaned_content = re.sub('】', ']', cleaned_content)

    # g16 CleanPersonInfoDoc
    cleaned_content = cleaner.clean_private(cleaned_content)

    # g6 CleanURL
    cleaned_content = cleaner.clean_url(cleaned_content)

    # g7 CleanContinueousPuncs, not used for cn-wiki dataset
    # cleaned_content = cleaner.clean_continueous_punc(cleaned_content)

    # g10 CleanDuplicationInText
    cleaned_content = cleaner.delete_2repeating_long_patterns(cleaned_content)
    cleaned_content = cleaner.deleteSpaceBetweenChinese(cleaned_content)
    # print(cleaned_content)
    # g13 TooShortSentence
    cleaned_content = cleaner.filter_long_sentences(cleaned_content)
    # print(cleaned_content)
    # g20 TooLongSentence
    cleaned_content = cleaner.remove_long_strings_without_punctuation(cleaned_content)
   
    # special usage

    # g24 LongEnough
    if len(cleaned_content) < _TEXT_LONG_REQUIRED_:
        return "TooShortSentence", 0, cleaned_content
    return "step1", 1, cleaned_content

def step2(text):
    cleaner.remove_strings_with_keywords(text, sensitive_words)

def step3():
    pass

def controller(args):
    files = sorted(listdir(args.source_path))
    for line_no,input_file in tqdm(enumerate(files),total=len(files)):
        input_file = input_file.strip()
        if len(input_file) < 1: continue
        yield (args,input_file)

def make_clean(items):
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
        for line_no,line in tqdm(enumerate(f.readlines())):
            line = line.strip()
            if len(line) < 1: continue
            
            js_dict = json.loads(line)
            content = js_dict["content"]

    good_output_file = os.path.join(args.dest_path,"good",input_file)
    if os.path.exists(good_output_file): os.remove(good_output_file)
    good_fo = open(good_output_file, 'a+', encoding='utf-8')

    bad_output_file = os.path.join(args.dest_path,"bad",input_file)
    if os.path.exists(bad_output_file): os.remove(bad_output_file)
    bad_fo = open(bad_output_file, 'a+', encoding='utf-8')

    fpath = os.path.join(args.source_path,input_file)
    with open(fpath,"r",encoding="utf-8") as f:
        for line_no,line in tqdm(enumerate(f.readlines())):
            line = line.strip()
            if len(line) < 1: continue
            
            js_dict = json.loads(line)
            content = js_dict["content"]

    # step 1: text normalization
    clean_policy,clean_status,text = step1(line)
    if clean_status == 1:
        # step 2: low quality
        clean_policy,clean_status,text = step2(text)
        if clean_status == 1:
            # step 3: text duplication
            clean_policy,clean_status,text = step3(text)

            if clean_status == 1: good_fo.write(jstr+"\n")
            else: bad_fo.write(jstr+"\n")
            
            if line_no % _FLUSH_STEPS_NUM_ == 0:
                good_fo.flush()
                bad_fo.flush()
    good_fo.close()
    bad_fo.close()

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

    pools = mp.Pool(_PROCESS_)
    pools.imap(make_clean, controller(args))
 
