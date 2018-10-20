#!/bin/bash
code=./scripts/preprocessing/gen_img_mini_batches.py
exe=~/tf18_p3_gpu/bin/python3

run_script="$exe $code"
$run_script
