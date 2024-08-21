# -*- encoding:utf-8 -*-
import os
import re
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SpecialPolicies():
    def __init__(self,):
        pass

    @staticmethod
    def IsChatperText(text,threashold=10,thresh_ratio=0.25):
        
        if len(text) < 1: return False

        first_num = len(re.findall(r'第[0-9一二三四五六七八九十百千万壹贰叁肆伍陆柒捌玖拾佰仟]+章', text))
        second_num = len(re.findall(r'（[0-9一二三四五六七八九十百千万壹贰叁肆伍陆柒捌玖拾佰仟]+）', text))
        
        if first_num > 10 or second_num > 10: return True

        first_num = 0
        second_num = 0
        for item in re.findall(r'第[0-9一二三四五六七八九十百千万壹贰叁肆伍陆柒捌玖拾佰仟]+章', text):
            first_num += len(item)
        for item in re.findall(r'（[0-9一二三四五六七八九十百千万壹贰叁肆伍陆柒捌玖拾佰仟]+）', text):
            second_num += len(item)

        frist_ratio = 1.0*first_num / len(text)
        second_ratio = 1.0*second_num / len(text)
        
        if frist_ratio > thresh_ratio or second_ratio > thresh_ratio: return True

        return False

    @staticmethod
    def RemoveReference(text):
        # 这种行为不合情理pp=66–67, 70。
        # 开创了塞萨洛尼基王国pp=62–63。

        #text = re.sub(r"p{1,2}=\d+–*\d*,*\s*\d*–*\d*","",text)
        regex = re.compile(r"p{1,2}=(\d+–*\d*,*\s*)+")
        text = regex.sub("",text)
        return text

    @staticmethod
    def RemoveLastLineBreak(text):
        text = text.strip().strip("\n").strip()
        return text

    @staticmethod
    def RemoveHeadWords(text):
        head_words = ["概述","图片发自简书app","[转载]"]
        for item in head_words:
            text = text.lstrip(item)
        text = text.strip()
        return text
 
    
    @staticmethod
    def RemoveSpamFromContent(text,spam):
        regex = re.compile(spam)
        text = regex.sub("",text)
        text = text.strip()
        return text

    @staticmethod
    def RemoveAllReference(text):
        # 参考文献:
        regex = re.compile(r"参考文献[:：].*")
        text = regex.sub("",text)
        text = text.strip()
        return text

    @staticmethod
    def delete_like_collect_comment(sentence):
        '''
        For ods_zdm_detail, match and remove 点赞 收藏 评论
        '''
        like = re.compile(r'\d*点赞')
        collect = re.compile('\d*收藏')
        comment = re.compile(r'\d*评论')
        sent = like.sub('', sentence)
        sent = collect.sub('', sent)
        sent = comment.sub('', sent)
        return sent

    @staticmethod
    def delete_author_claim(sentence):
        '''
        For ods_zdm_detail, match and remove 作者声明xxxx
        '''
        pattern = re.compile(r'作者声明.*|本文商品由什么.*|小编注.*|以上是.*分享.*|全文完.*|(感谢|谢谢).*(众测|测评|机会|值友).*|我是.*|(链接|商品链接).*?(去购买|去看看)|未经授权，不得转载.*|本文[^。]*.$|\|赞\d.*|The.{0,1}End.*')
        return pattern.sub('', sentence)

    @staticmethod
    def detect_lottery(sentence):
        '''
        For ods_zdm 
        '''
        pattern = re.compile(r'(获奖|有奖).*活动')
        if pattern.search(sentence):
            return False
        else:
            return sentence

    # 2023-08-16
    @staticmethod
    def RemovewechatID(text):
        # 参考文献:
        regex = re.compile(r"微信.{0,5}[a-zA-Z_][-_a-zA-Z0-9]{5,19}")
        text = regex.sub("",text)
        text = text.strip()
        return text        
    @staticmethod
    def RemoveAllUnicode(text):
        # Unicode 编码 like <200a> <200b>:
        regex = re.compile(r"<[0-f]{4}>")
        text = regex.sub("",text)
        text = text.strip()
        return text
    
    @staticmethod
    def is_mixed_ENCN(text):
        def is_chinese(char) -> bool:
            return char.isdigit() or ('\u4e00' <= char <= '\u9fa5') or char in ['\u3002','\uff1b','\uff0c','\uff1a','\u201c','\u201d','\uff08','\uff09','\u3001','\uff1f','\u300a','\u300b']
        def is_mixed_seq(seq):
            sub = np.array(list(map(is_chinese, seq)), dtype=int)
            if np.sum(np.abs(sub[1:]-sub[:-1]))>=6:
                # print('ilegal:', seq)
                return True
            return False
        sample_num = 10 if len(text)>70 else len(text)//10
        starts = np.linspace(0, len(text)-7, num=sample_num, endpoint=True, dtype=int)
        for start in starts:
            if is_mixed_seq(text[start:start+7]):
                return True
        return False


    
