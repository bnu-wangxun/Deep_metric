#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python train.py -data cub -r checkpoints/bn_cub_neig_1e5/800_model.pkl -start 800  -loss neighbour -net bn -BatchSiz 150 -num_instances 6 -epochs 1600 -lr 1e-5 -log_dir bn_cub_neig_1e5

