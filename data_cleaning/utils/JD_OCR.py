# -*- coding:utf-8 -*-
# @Time       :2023/5/5 10:59
# @AUTHOR     :YUNYI
# @SOFTWARE   :instruct-data-clean
# @DESC       : 读取小红书数据

from DorisTools import DorisLogger, DorisSession
from xToolkit import xfile
import os
import sys
import json
import hashlib
import re
import multiprocessing as mp
import argparse
from os import listdir
from tqdm import tqdm

PROCESS = 16


class read_data(object):
    def __init__(self):
        """
        初始化链接信息
        """
        config = {
            'fe_servers': ['10.200.2.103:9030'],
            'database': 'dwd_jd',
            'user': 'shenxuyang',
            'passwd': 'Frontis2021xy',
            'prot': 9030,
            'charset': 'utf8'
        }
        self.client = DorisSession(doris_config=config)
        self.logger = DorisLogger

    def get_jd_note(self, item_id: str):
        """
        查询指定日期小红书数据：date 2023-04-01
        返回格式：[{},{}]
        """
        self.logger.info("开始执行item_id: {}的数据".format(item_id))
        sql = """
        select item_id, title, params, cat_name, current_package from dwd_jd.dwd_jd_detail 
        where item_id = '{id}'
        group by item_id, title, params, cat_name, current_package
        """.format(id=int(item_id))
        res = self.client.select(sql)
        return res

read = read_data()

def extract_sentences_with_colon(text):
    sentence_delimiters = r'[,.，。；！？]'
    sentences = re.split(sentence_delimiters, text)
    extracted_sentences = {}
    remaining_text = ""

    for sentence in sentences:
        # 判断输入字符串是否包含":"或"："
        if ":" in sentence or "：" in sentence:
            # 以":"或"："为分隔符进行拆分
            key, value = sentence.split(":" if ":" in sentence else "：", 1)
            extracted_sentences[key.strip()] = value.strip()
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
        if ans != "": ans = ""
        ans += content
    return ans

def extract(input):
    item_id, ocr_text = input
    pairs, text = {}, ''
    if len(ocr_text) > 10:
        pairs, text = extract_sentences_with_colon(ocr_text)
    output = {
        "item_id": item_id,
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
            jd_data = read.get_jd_note(res['item_id'])
            new_res = {
            'item_id': res['item_id'],
            'source': 'OCR+JD_TEXT',
            'source_id': '',
            'content': {
                'pairs': res['content']['pairs']|json.loads(jd_data[0]['params']),
                'text': res['content']['text'] + ',' + jd_data[0]['title'] + ',' + jd_data[0]['cat_name'] +','+ jd_data[0]['current_package'],
                'qa': ''
                }
            }
            #print(output)
            jstr = json.dumps(new_res, ensure_ascii=False)
            print(jstr)
            output.write(jstr+"\n")
            flush_steps += 1
            if flush_steps % flush_per_steps == 0:
                output.flush()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path',
                        type=str,
                        default="/hpc_data/data_warehouse/llm/source_data/cn-JD-ocrtext",
                        help='OCR PATH')
    parser.add_argument('--dest_path',
                        type=str,
                        default="/data/jiangdingyi/test/ocrtest",
                        help='OCR STORE PATH')
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

    
