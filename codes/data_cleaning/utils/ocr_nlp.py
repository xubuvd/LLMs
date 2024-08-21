# -*- coding: utf-8 -*-
import os
import sys
import json
import hashlib
import re
import multiprocessing as mp
import argparse
from os import listdir
from tqdm import tqdm
from util import load_set_from_txt

PROCESS = 2

def extract_sentences_with_colon(text):
    sentence_delimiters = r'[,.，。；！？]'
    sentences = re.split(sentence_delimiters, text)
    extracted_sentences = []
    remaining_text = ""

    for sentence in sentences:
        if ':' in sentence or '：' in sentence:
            extracted_sentences.append(sentence.strip())
        else:
            remaining_text += sentence.strip() + " "

    return extracted_sentences, remaining_text.strip()

def controller(input_file):
    for line in open(input_file,'r',encoding="utf-8"):
        line = line.strip()
        if len(line) < 5:continue

        js_dict = json.loads(line)
        item_id = js_dict["item_id"]
        ocr_ret_list = js_dict["ocr_ret"]
        ocr_text = ""
        for item in ocr_ret_list:
            image_name = item["img_name"]
            ocr_ret = item["ocr_ret"]
            one_img_content = concat_one_img(ocr_ret)
            if len(one_img_content) < 5:continue
            if ocr_text != "": ocr_text += "。"
            ocr_text += one_img_content
        yield item_id, ocr_text

#reload(sys)
#sys.setdefaultencoding('utf-8')

'''
{"ocr_ret": [{"img_name": "/vmware_data/gaodiqi/jingdong_imgs/100000040875/detailimg_e6d026e0d93c15d7174425f9c778eb7e.jpg", "ocr_ret": [{"index": [[248.0, 172.0], [619.0, 172.0], [619.0, 217.0], [248.0, 217.0]], "content": "      年风霜，匠心如初", "confidence": 0.9926699995994568}, {"index": [[58.0, 284.0], [741.0, 283.0], [741.0, 324.0], [58.0, 325.0]], "content": "品质依然，福东海健康食材的选择", "confidence": 0.9772330522537231}, {"index": [[246.      0, 351.0], [306.0, 351.0], [306.0, 371.0], [246.0, 371.0]], "content": "黄民", "confidence": 0.8339320421218872}, {"index": [[406.0, 352.0], [470.0, 349.0], [471.0, 369.0], [407.0, 372.0]], "content": "胎菊", "confidence": 0      .8963175415992737}, {"index": [[571.0, 351.0], [631.0, 351.0], [631.0, 371.0], [571.0, 371.0]], "content": "贡菊", "confidence": 0.8061279058456421},
'''
def concat_one_img(ocr_ret_list):

    ans = ""
    duplicate_set = set()
    for item in ocr_ret_list:
        index = item["index"]
        content = item["content"].strip()
        if len(content) < 1:continue
        md5 = hashlib.md5(content.encode('utf-8')).hexdigest()
        #print("md5:",md5)
        if md5 in duplicate_set: continue
        duplicate_set.add(md5)
        if ans != "": ans = "，"
        ans += content
    return ans

def extract(input):
    item_id, ocr_text = input
    pairs, text = [], ''
    if len(ocr_text) > 10:
        pairs, text = extract_sentences_with_colon(ocr_text)
    output = {
        "id": item_id,
        "source": "OCR",
        "source_id":"",
        "content": {"pairs": pairs,"text": text, "qa":""}
    }
    return output
    
def HandleSingleFile(input_file, output):
    pools = mp.Pool(PROCESS)

    flush_steps = 0
    flush_per_steps = 50
    for res in pools.imap(extract, controller(input_file)):
        if res is not None:
            jstr = json.dumps(res, ensure_ascii=False)
            output.write(jstr+"\n")
            flush_steps += 1
            if flush_steps % flush_per_steps == 0:
                output.flush()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path',
                        type=str,
                        default="/root/llm/source_data/cn-JD-ocrtext/",
                        help='Directory containing trained actor model')
    parser.add_argument('--dest_path',
                        type=str,
                        default="/root/llm/clean_data/cn-JD-ocrtext/",
                        help='Directory containing trained actor model')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    files = sorted(listdir(args.source_path))

    Output_Dir = os.path.join(args.dest_path)

    if not os.path.exists(Output_Dir):
        os.makedirs(Output_Dir, exist_ok=True)

    for input_file in tqdm(files,total=len(files)):
        input = os.path.join(args.source_path, input_file)
        output_file = os.path.join(Output_Dir, input_file)
        if os.path.exists(output_file): os.remove(output_file)
        output = open(output_file, 'a+', encoding='utf-8')

        HandleSingleFile(input, output)

        output.close()

