#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

code=odes/experiments/run_inference.py 
config=odes/configs/vgg_cars.config
gpu='1'
data='test' #'val'
ckpt_indices=0
checkpoint_name=vgg_cars
exe=~/tf18_p3_gpu/bin/python3
#exe=~/tf18_p3_gpu/bin/pudb3


run_script="$exe $code --device=$gpu --data_split=$data --ckpt_indices=$ckpt_indices --checkpoint_name=$checkpoint_name"
$run_script


