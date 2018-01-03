#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=7 python train.py -data products -net bn  -lr 1e-5 -num_instances 3 -BatchSize 126  -loss neighbour -epochs 401 -log_dir products_m_01_1e5_n_8_b_128

