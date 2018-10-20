#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

code=odes/experiments/run_evaluation.py 
#config=odes/configs/resnet_cars.config
config=odes/configs/vgg_cars.config
gpu='0'
data='val'
exe=~/tf18_p3_gpu/bin/python3
#exe=~/tf18_p3_gpu/bin/pudb3

run_script="$exe $code --pipeline_config=$config --device=$gpu --data_split=$data"
$run_script


