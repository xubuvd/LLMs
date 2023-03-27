#!/bin/bash

#ulimit -c unlimited
#ulimit -n 8192
for (( c=1; c<=1; c++ ))
do
    echo "Welcome $c times"
    CUDA_VISIBLE_DEVICES='1,2,3' python train.py
    if [ $? == 0 ]; then
        echo "Successed at $c times"
        break
    fi
done

