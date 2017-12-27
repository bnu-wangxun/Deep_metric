#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=7 python train.py -data cub -net bn  -lr 1e-5 -num_instances 8 -BatchSize 128  -loss neighbour -epochs 3000 -log_dir cub_adam_1e5_n_8_b_128

