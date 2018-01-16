#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train.py -data products -net bn  -lr 1e-5 -num_instances 4 -BatchSize 128  -loss neighbour -epochs 301 -log_dir products_m_01_1e5_n_4_b_128  -step 30 

