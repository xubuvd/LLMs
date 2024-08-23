#!/bin/bash
ps aux|grep postrain_with_dpo.sh |grep -v grep | awk '{print $2}' | while read pid ;do  kill -9 $pid;done;
ps aux|grep postrain.py |grep -v grep | awk '{print $2}' | while read pid ;do  kill -9 $pid;done;

