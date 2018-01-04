#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=7 python train.py -data products -net bn  -lr 1e-5 -num_instances 4 -BatchSize 128  -loss neighbour -epochs 501 -log_dir products_m_01_1e5_n_4_b_128 -r checkpoints/products_m_01_1e5_n_4_b_128/60_model.pkl

