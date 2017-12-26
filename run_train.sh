#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python train.py -data cub  -loss neighard -net bn -BatchSiz 150 -num_instances 6 -epochs 1600 -lr 1e-5 -log_dir bn_cub_neighard_1e5

