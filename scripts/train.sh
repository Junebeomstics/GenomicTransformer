#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

CUDA_VISIBLE_DEVICES=3 python lm_main.py \
    --dataset wiki103 \
    --loss-type plain \
    --root ../data/emnlp \
    --encoder-class SPBPE \
    --hierarchical \
    --vocab-size 30000;
#    --hierarchical \
