# _*_coding:utf-8 _*_
import json
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import fasttext 
from tqdm import tqdm

#加载模型
model = fasttext.load_model('./fastText_shortAd/models/fasttext_train.model.bin')

labels_right = []
texts = []
labels_predict = []

with open("/data/data_warehouse/llm/source_data/cn-wechat/wx_data_980.jsonl") as fr:
    datas = fr.readlines()
    for idx,line in tqdm(enumerate(datas),total=len(datas)):
        line = line.strip()
        if len(line) < 5: continue
        js_dict = json.loads(line)
        text = js_dict["content"].strip()
        text = text.replace("\n"," ")
        label_predict = model.predict(text)
        labels_predict.append(label_predict[0])
        print ("文本: ",text[0:200])
        print ("预测label: ",label_predict[0])
        print("-"*60)

