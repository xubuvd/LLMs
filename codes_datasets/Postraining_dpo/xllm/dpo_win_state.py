import argparse
import json
import os
import time
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="./dpo_ckpt_70b_32k_v6-sft-Llama3-70b_final_4epoch-frontis.json",
        help=""
    )
    parser.add_argument(
        "--win_var",
        type=int,
        default=4,
        help=""
    )
    args = parser.parse_args()
    
    js_dict_list = json.load(open(args.input_file))
    total_win_var_num = 0
    total_loss_var_num = 0
    for idx,js_dict in tqdm(enumerate(js_dict_list),total=len(js_dict_list)):
        score_list = js_dict['score']
        dpo_score = int(score_list[0])
        sft_core = int(score_list[1])

        if dpo_score >= 10 - args.win_var and sft_core <= args.win_var:
            total_win_var_num += 1
        if sft_core >= 10 - args.win_var and dpo_score <= args.win_var:
            total_loss_var_num += 1
    print("total numbers of DPO eval dataset: ", len(js_dict_list))
    print("total significantly improved instances that below {} score: {}".format(args.win_var,total_win_var_num))
    print("total significantly declined instances that below {} score: {}".format(args.win_var,total_loss_var_num))
    print("significantly improved ratio: {}%".format(100.0*total_win_var_num/len(js_dict_list)))
    print("significantly declined ratio: {}%".format(100.0*total_loss_var_num/len(js_dict_list)))

