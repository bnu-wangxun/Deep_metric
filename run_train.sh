#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python train.py -data products -net bn  -lr 1e-5 -num_instances 4 -BatchSize 152  -loss neighbour -epochs 401 -log_dir products_m_01_1e5_n_8_b_152  -step 20 

