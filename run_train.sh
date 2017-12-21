#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train.py -data cub -BatchSize 128 -num_instances 4 -epochs 1000 -lr 1e-5
