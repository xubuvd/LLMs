# -*- encoding:utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flashtext import KeywordProcessor
from utils.util import load_list_from_structedTxt

class CleanHeadTailsFromContent:
    def __init__(self, keyphrase_file, thresh_hold=5):
        self.ads_wechat_flashtext = KeywordProcessor()

        ads_phrase_list = load_list_from_structedTxt(keyphrase_file)
        print(f"load {len(ads_phrase_list)} ads phrases:",ads_phrase_list)
        self.ads_wechat_flashtext.add_keywords_from_list(ads_phrase_list)
        self.split_flg = ['。','\n','！','？']
        self.thresh_hold = thresh_hold

    def clean(self,text):
        text = text.strip()
        text = self.forward(text)
        #print("forward:",text)
        text = self.backward(text)
        #print("backward:",text)
        return text

    def forward(self,text):
        
        prev_density = 0.0
        prev_idx = 0

        no_hit_sentence_cnt = 0
        hit_pos = 0

        tlen = len(text)
        while prev_idx < tlen:
            curr_idx = prev_idx
            while curr_idx < tlen and text[curr_idx] not in self.split_flg: curr_idx += 1

            head_text = text[prev_idx:curr_idx+1]
            if len(head_text) < 1:
                prev_idx = curr_idx + 1
                continue
            diff_cnt,keylen = self.calculate_density(head_text)
            #print(f"head_text:{head_text}, diff_cnt:{diff_cnt}, keylen:{keylen}")
            if diff_cnt < 1:
                prev_idx = curr_idx + 1
                no_hit_sentence_cnt += 1
                if no_hit_sentence_cnt >= self.thresh_hold: break
            else:
                hit_pos = curr_idx + 1
                prev_idx = curr_idx + 1
                no_hit_sentence_cnt = 0
        text = text[hit_pos:].strip()
        return text

    def backward(self,text):
        last_density = 0.0
        last_idx = len(text) - 1

        no_hit_sentence_cnt = 0
        hit_pos = last_idx

        while last_idx > 0:
            curr_idx = last_idx
            while curr_idx > 0 and text[curr_idx] not in self.split_flg: curr_idx -= 1

            tail_text = text[curr_idx+1:last_idx+1]
            if len(tail_text) < 1:
                last_idx = curr_idx - 1
                continue
            diff_cnt,keylen = self.calculate_density(tail_text)
            #print(f"tail_text:{tail_text}, diff_cnt:{diff_cnt}, keylen:{keylen}")
            if diff_cnt < 1:
                last_idx = curr_idx
                no_hit_sentence_cnt += 1
                if no_hit_sentence_cnt >= self.thresh_hold: break
            else:
                hit_pos = curr_idx
                last_idx = curr_idx
                no_hit_sentence_cnt = 0
        text = text[0:hit_pos+1].strip()
        return text

    def calculate_density(self,text):
        keyword_list = self.ads_wechat_flashtext.extract_keywords(text)
        #print("keyword_list:",keyword_list)
        keylen = sum([len(item) for item in keyword_list])
        #ratio = 1.0*keylen / (len(text) + 0.005)
        diff_cnt = len(set(keyword_list))
        return diff_cnt,keylen


