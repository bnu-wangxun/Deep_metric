# coding=utf-8
from __future__ import absolute_import, print_function
import argparse

import torch
from torch.backends import cudnn
import os.path as osp
from evaluations import extract_features, pairwise_distance
from evaluations import Recall_at_1
import DataSet

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='PyTorch Testing')

parser.add_argument('-data', type=str, default='car')
parser.add_argument('-r', type=str, default='model.pkl', metavar='PATH')

parser.add_argument('-test', type=bool, default=True, help='evaluation on test set or train set')

args = parser.parse_args()
# torch.cuda.set_device(8)
model_path = osp.join('checkpoint', args.r)
cudnn.benchmark = True

# model = inception_v3(dropout=0.5)
model = torch.load(model_path)
model = model.cuda()

if args.test:
    data = DataSet.create(args.data, train=False)
    data_loader = torch.utils.data.DataLoader(
        data.test, batch_size=64, shuffle=False, drop_last=False)
else:
    data = DataSet.create(args.data, test=False)
    data_loader = torch.utils.data.DataLoader(
        data.train, batch_size=64, shuffle=False, drop_last=False)

features, labels = extract_features(model, data_loader, print_freq=1, metric=None)

# print(len(features))
dist_mat = pairwise_distance(features)
print(Recall_at_1(dist_mat, query_ids=labels, gallery_ids=labels))
