
import re
import numpy as np
import json
import random
import os
import hashlib


'''
{'fldStatus': 2, 'fldColumnID': 4, 'fldSubject': '蓝点Linux半年上市', 'fldContent': '','fldCreateTime': '2000-04-20 18:03:59', 'fldColumnName': '中国.com', 'fldUserID': 'liuren', 'fldName': '刘韧', 'fldView': 472, 'fldTypeID': '原创-IT', 'fldArticleID': 4, 'fldUserNum': 2}
'''
def json2jsonl():
    with open("./donews.18402.json","r",encoding='utf-8') as fo: data = json.load(fo)

    sft_data = []
    idx = 0
    for idx,item in enumerate(data):
        js_dict = {}
        js_dict["id"] = idx + 1
        js_dict["source"] = "donews"
        js_dict["subset"] = item["fldSubject"]
        js_dict["source_id"] = ""
        js_dict["fldCreateTime"] = item["fldCreateTime"]
        js_dict["fldTypeID"] = item["fldTypeID"]
        js_dict["content"] = item["fldContent"]
        sft_data.append(js_dict)

    dest_file = os.path.join("./","donews.18402.jsonl")
    if os.path.exists(dest_file): os.remove(dest_file)
    of = open(dest_file,'w',encoding='utf-8')

    #random.shuffle(sft_data)
    for item in sft_data:
        print(json.dumps(item,ensure_ascii=False),file=of)
    of.close()
    print(f"writting {len(sft_data)} lines into {dest_file}")

def data2mnbvc_style(input_file,output_dir):
    '''
    {
        '文件名': '文件.txt',
        '是否待查文件': False,
        '是否重复文件': False,
        '文件大小': 1024,
        'simhash': 0,
        '最长段落长度': 0,
        '段落数': 0,
        '去重段落数': 0,
        '低质量段落数': 0,
        '段落': [
            {
                '行号': 1,
                '是否重复': False,
                '是否跨文件重复': False,
                'md5': 'md5hash1',
                '内容': '这是第一段文字。'
            }
        ]
    }
    '''
    global_file_no = 0
    dest_file = os.path.join(output_dir,"mnbvc-donews-part-{:06d}.jsonl".format(global_file_no))
    if os.path.exists(dest_file): os.remove(dest_file)
    of = open(dest_file,'w',encoding='utf-8')
   
    for line in open(input_file,"r",encoding='utf-8'):
        line = line.strip()
        if len(line) < 5: continue
        js_dict = json.loads(line)

        js_new = {}
        js_new['文件名'] = js_dict["source"]
        js_new['是否待查文件'] = False
        js_new['是否重复文件'] = False
        js_new['文件大小'] = len(js_dict["content"])
        js_new['simhash'] = ''
        js_new['最长段落长度'] = len(js_dict["content"])
        js_new['段落数'] = 1
        js_new['去重段落数'] = 0
        js_new['低质量段落数'] = 0
        js_new['段落'] = []

        item = {}
        item['行号'] = 1
        item['是否重复'] = False
        item['是否跨文件重复'] = False

        content = js_dict["content"]
        md5 = hashlib.md5(content.encode('utf-8')).hexdigest()
        item['md5'] = md5
        item['内容'] = js_dict["content"]
        js_new['段落'].append(item)

        print(json.dumps(js_new,ensure_ascii=False),file=of)
        if of.tell() > 20 * 1024 * 1024:
            of.close()
            dest_file = os.path.join(output_dir,"mnbvc-donews-part-{:06d}.jsonl".format(global_file_no))
            if os.path.exists(dest_file): os.remove(dest_file)
            of = open(dest_file,'w',encoding='utf-8')
            global_file_no += 1

    of.close()


if __name__ == "__main__":
    input_file = "../llm/clean_data/cn-donews/v1/good/donews.18402.jsonl" 
    output_file = "../llm/clean_data/cn-donews/v1/good/"
    data2mnbvc_style(input_file,output_file)

