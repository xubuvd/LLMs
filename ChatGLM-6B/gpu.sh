#!/bin/bash

id=$1
#gpustat
#watch --color -n2 nvidia-smi #-i ${id}
watch --color -n2 gpustat -cpu # -i ${id}

#watch --color -n2 gpustat -cpu 
#watch -n 2 nvidia-smi
#nvidia-smi -l 2

#执行fuser -v /dev/nvidia* 发现僵尸进程（连号的）
#fuser -v /dev/nvidia*

#find / -name '*.fcitx' | xargs  rm -rf
#ps -ef | grep firefox | grep -v grep | cut -c 9-15 | xargs kill -9
#ps -ef | grep "train.py" | grep -v grep | awk '{print $2}' | xargs kill -9

# 总核数 = 物理CPU个数 X 每颗物理CPU的核数 
# 总逻辑CPU数 = 物理CPU个数 X 每颗物理CPU的核数 X 超线程数
# 查看物理CPU个数
#cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l
# 查看每个物理CPU中core的个数(即核数)
#cat /proc/cpuinfo| grep "cpu cores"| uniq
# 查看逻辑CPU的个数
#cat /proc/cpuinfo| grep "processor"| wc -l
#查看CPU信息（型号）
#cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq -c

#把内存当硬盘
# sudo mount tmpfs /home/xubuvd/visdial-rl/data/visdial -t tmpfs -o size=64G
