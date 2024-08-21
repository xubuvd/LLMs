# -*- coding: utf-8 -*-

import fasttext
import os
import jieba
import json
import tqdm
import argparse
from multiprocessing import Pool
from functools import partial
# import gensim


def train(train_file, word2vec_file, model_file):
    ### Train the FastText classifier using the pre-trained embeddings
    if word2vec_file:
        model = fasttext.train_supervised(input=train_file, dim=300, label_prefix="__label__", epoch=25, lr=1.0, wordNgrams=3, pretrainedVectors=word2vec_file)
    else:
        model = fasttext.train_supervised(input=train_file, dim=300, label_prefix="__label__", epoch=25, lr=1.0, wordNgrams=3)

    ### Save the trained model
    model.save_model(model_file)


def test(test_file, model_file):
    if os.path.exists(model_file):
        model = fasttext.load_model(model_file)
    else:
        raise
    print(model.test(test_file))


def predict(predict_file, model_file, stop_words):
    with open(predict_file) as f:
        text_list = [line.strip() for line in f]
    predicted_label, _ = predict_list(text_list, model_file, stop_words)
    res = list([f"{i} {label[0][9:]} {float(score[0]):.2f}" for i, (label, score) in enumerate(zip(predicted_label[0], predicted_label[-1]))])
    print(res)


def predict_list(text_list, model_file, stop_words):
    model = fasttext.load_model(model_file)
    seg_list = []
    for text in text_list:
        seg_text = jieba_cut(text, stop_words)
        seg_list.append(seg_text)
    predicted_label = model.predict(seg_list)
    return predicted_label, text_list


def jieba_cut(text, stop_words):
    seg_text = jieba.cut(text)
    seg_text_clean = [word.strip() for word in seg_text if word not in stop_words]
    text = " ".join(seg_text_clean)
    text.replace("\n", " ")
    return text


def read_stop_words(stop_word_path):
    stop_words = set()
    with open(stop_word_path) as f:
        for line in f:
            line = line.strip()
            stop_words.add(line)
    return stop_words


def jsonl_loader(file_path, batch=1024):
    with open(file_path) as f:
        tmp = []
        for line in f:
            line = line.strip()
            text = json.loads(line)["text"]
            tmp.append(text)
            if len(tmp) >= batch:
                yield tmp
                tmp = []
        if tmp:
            yield tmp

def write_jsonl(f1, f2, res, text_list):
    for label, score, text in zip(res[0], res[-1], text_list):
        label = label[0][9:]
        json_dict = {"text": text, "score": float(score[0]), "label": label}
        line = json.dumps(json_dict) + "\n"
        if label == "ad":
            f2.write(line)
        elif label == "not_ad":
            f1.write(line)

def clean_vis(file_path, model_file, stop_words, out_path, batch=1024):
    num_process = 10
    batch = 1024

    # 如果输出目录不存在，则创建
    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    path_, ext = os.path.splitext(out_path)
    label_0_out_path, label_1_out_path = path_ + "_0" + ext, path_ + "_1" + ext
    with open(label_0_out_path, "w") as f1, open(label_1_out_path, "w") as f2:
        loader = jsonl_loader(file_path, batch)
        predict = partial(predict_list, model_file=model_file, stop_words=stop_words)
        if num_process <= 1:
            for text in loader:
                try:
                    res, text_list = predict(text)
                    write_jsonl(f1, f2, res, text_list)
                except:
                    print(text)
        else:
            idx = 0
            with Pool(num_process) as pool:
                for res, text_list in pool.imap_unordered(predict, loader):
                    write_jsonl(f1, f2, res, text_list)
                    idx += batch
                    print(f"finish {idx}")

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-I', '--input_file', type=str, help='input file')
    parser.add_argument('--pretrain_w2v', default="./w2v/cc.zh.300.vec", type=str, help='pretain w2v to load when training')
    parser.add_argument('--model_file', default="./output_models/fasttext.bin", type=str, help='output model path')
    parser.add_argument('--stop_word_path', default="./data/stop_words.txt", type=str, help='stop words to drop')
    parser.add_argument('--output_path', default="./cc_clean/clean_res.jsonl", type=str, help='stop words to drop')
    parser.add_argument('-M', '--mode', type=str, default="train", help='train: train, test:test, predict:predict')
    
    args = parser.parse_args()

    stop_word_path = args.stop_word_path
    stop_words = read_stop_words(stop_word_path)
    input_file = args.input_file
    word2vec_file = args.pretrain_w2v
    model_file = args.model_file
    out_path = args.output_path
    mode = args.mode

    if mode == "train":
        train(input_file, word2vec_file, model_file)

    if mode == "test":
        test(input_file, model_file)

    if mode == "predict":
        predict(input_file, model_file, stop_words)

    if mode == "clean":
        clean_vis(input_file, model_file, stop_words, out_path, batch=1024)

if __name__ == "__main__":
    main()

