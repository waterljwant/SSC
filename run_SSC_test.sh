#!/usr/bin/env bash

source /home/amax/jie/pyenv/pytorch1.2/bin/activate


CUDA_VISIBLE_DEVICES=0,1,2,3 python ./test.py \
--model='ddrnet' \
--dataset=nyucad \
--batch_size=4 \
--resume='path\to\pretrained\weights' 2>&1 |tee test_DDRNet_NYUCAD.log


deactivate

