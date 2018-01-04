#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=8 python train.py -data car -net bn  -lr 1e-5 -num_instances 8 -BatchSize 128  -loss neighbour -epochs 1201 -log_dir car_m_01_1e5_n_8_b_128  -step 100 

