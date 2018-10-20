#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

code=odes/experiments/run_training.py 
config=odes/configs/vgg_cars.config
gpu='1'
data='train'
#exe=~/tf18_p3_gpu/bin/python3
exe=~/tf18_p3_gpu/bin/pudb3


run_script="$exe $code --pipeline_config=$config --device=$gpu --data_split=$data"
$run_script
