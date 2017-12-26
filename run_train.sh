#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train.py -data cub -loss distance_match -BatchSize 128 -num_instances 8 -epochs 3000 -lr 1e-5 -log_dir cub_dist_1e-5
