#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train.py -data cub -loss neighbour -net bn -BatchSiz 126 -num_instances 6 -epochs 3000 -lr 1e-4 -log_dir bn_cub_neig_1e4
