#!/bin/bash
ps aux|grep wanjuan_clean.py |grep -v grep | awk '{print $2}' | while read pid ;do  kill -9 $pid;done;
ps aux|grep tokenizer.py |grep -v grep | awk '{print $2}' | while read pid ;do  kill -9 $pid;done;
ps aux|grep random_sample.py |grep -v grep | awk '{print $2}' | while read pid ;do  kill -9 $pid;done;
