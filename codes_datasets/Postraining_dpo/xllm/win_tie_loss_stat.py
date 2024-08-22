import argparse
import json
import os
import time
#import openai
from tqdm import tqdm
import asyncio
from typing import Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_score(review):
    
    if isinstance(review,list): return review
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            raise Exception("Invalid score pair.")
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return [-1, -1]

def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    parser.add_argument("-i1", "--input1_file")
    parser.add_argument("-i2", "--input2_file")
    parser.add_argument("-k1", "--key_1")
    parser.add_argument("-k2", "--key_2")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size to call OpenAI GPT",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="maximum number of tokens produced in the output",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The output dir."
    )
    parser.add_argument(
        "--dst",
        type=str,
    )
    args = parser.parse_args()
    
    i1_jsons = json.load(open(args.input1_file))
    i2_jsons = json.load(open(args.input2_file))
    assert len(i1_jsons) == len(i2_jsons)

    if "vicuna" in args.input1_file:
        prompt_key = 'text'
        data_idx = 'question_id'
        dst = 'vicuna' # dst is used for saving the content
    elif "koala" in args.input1_file:
        prompt_key = 'prompt'
        dst = 'koala'
        data_idx = 'id'
    elif "sinstruct" in args.input1_file:
        prompt_key = 'instruction'
        dst = 'sinstruct'
        data_idx = 'id'
    elif "wizardlm" in args.input1_file:
        prompt_key = 'Instruction'
        dst = 'wizardlm'
        data_idx = 'idx'
    elif "anthropic" in args.input1_file:
        prompt_key = 'instruction'
        dst = 'anthropic'
        data_idx = 'id'
    elif "oasst" in args.input1_file:
        prompt_key = 'instruction'
        dst = "oasst"
        data_idx = "id"
    elif "frontis" in args.input1_file:
        prompt_key = "data"
        dst = "frontis"
        data_idx = "id"

    total_len = len(i1_jsons)
    question_idx_list = list(range(total_len))
    
    k1_win_num = 0
    k1_tie_num = 0
    k1_loss_num = 0

    k2_win_num = 0
    k2_tie_num = 0
    k2_loss_num = 0

    for i in question_idx_list:
        assert i1_jsons[i][data_idx] == i2_jsons[i][data_idx]
        
        pos_order_score_pair = parse_score(i1_jsons[i]['score'])
        neg_order_score_pair = parse_score(i2_jsons[i]['score'])
        if not ( pos_order_score_pair[0] > 0 and pos_order_score_pair[1] > 0 and neg_order_score_pair[0] > 0 and neg_order_score_pair[1] > 0 ): continue
        
        if (pos_order_score_pair[0] > pos_order_score_pair[1] and neg_order_score_pair[1] > neg_order_score_pair[0]) or \
            (pos_order_score_pair[0] > pos_order_score_pair[1] and neg_order_score_pair[1] == neg_order_score_pair[0]) or \
            (pos_order_score_pair[0] == pos_order_score_pair[1] and neg_order_score_pair[1] > neg_order_score_pair[0]):
            k1_win_num += 1
        elif ( pos_order_score_pair[0] == pos_order_score_pair[1] and neg_order_score_pair[1] == neg_order_score_pair[0] ) or \
            (pos_order_score_pair[0] > pos_order_score_pair[1] and neg_order_score_pair[1] < neg_order_score_pair[0]) or \
            (pos_order_score_pair[0] < pos_order_score_pair[1] and neg_order_score_pair[1] > neg_order_score_pair[0]):
            k1_tie_num += 1
        elif ( pos_order_score_pair[0] < pos_order_score_pair[1] and neg_order_score_pair[1] < neg_order_score_pair[0]) or \
            ( pos_order_score_pair[0] < pos_order_score_pair[1] and neg_order_score_pair[1] == neg_order_score_pair[0] ) or \
            ( pos_order_score_pair[0] == pos_order_score_pair[1] and neg_order_score_pair[1] < neg_order_score_pair[0] ):
            k1_loss_num += 1
        else:
            print("ErrorLine...")
            assert False

    output_dir = args.output_dir
    output_review_file = args.key_1 + '-' + args.key_2 +'-'+ dst + '-comparison.json'
    if os.path.isdir(output_dir) is not True: os.mkdir(output_dir)
    output_review_f = os.path.join(output_dir, output_review_file)
    if os.path.exists(output_review_f): os.remove(output_review_f)
 
    with open(f"{output_review_f}", "w",encoding='utf-8') as fo:
        js_dict = {}

        js_dict["eval_scorer"] = "gpt-4-0613"
        js_dict["dataset"] = dst
        js_dict["num"] = total_len      
        js_dict[args.key_1] = {}
        js_dict[args.key_1]["win_rate"] = "{:+.2f}%".format(100.0*(k1_win_num + k1_tie_num/2.0)/total_len - 50.0)
        js_dict[args.key_1]["model1_name"] = args.key_1
        js_dict[args.key_1]["k1_win_num"] = k1_win_num
        js_dict[args.key_1]["k1_tie_num"] = k1_tie_num
        js_dict[args.key_1]["k1_loss_num"] = k1_loss_num
        
        js_dict[args.key_2] = {}
        js_dict[args.key_2]["win_rate"] = "{:+.2f}%".format(100.0*(k1_loss_num + k1_tie_num/2.0)/total_len - 50.0)
        js_dict[args.key_2]["model2_name"] = args.key_2
        js_dict[args.key_2]["k2_win_num"] = k1_loss_num
        js_dict[args.key_2]["k2_tie_num"] = k1_tie_num
        js_dict[args.key_2]["k2_loss_num"] = k1_win_num
        
        json.dump(js_dict, fo, indent=4)

