# coding=utf-8
from __future__ import absolute_import, print_function
import argparse

import torch
from torch.backends import cudnn
from evaluations import extract_features, pairwise_similarity
from evaluations import Recall_at_ks, NMI
import DataSet

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='PyTorch Testing')

parser.add_argument('-data', type=str, default='car')
parser.add_argument('-r', type=str, default='model.pkl', metavar='PATH')

parser.add_argument('-test', type=int, default=1, help='evaluation on test set or train set')

args = parser.parse_args()
cudnn.benchmark = True

# model = inception_v3(dropout=0.5)
model = torch.load(args.r)
model = model.cuda()

if args.test == 1:
    print('evaluation on test set of %s with model: %s' %(args.data, args.r))
    data = DataSet.create(args.data, train=False)
    data_loader = torch.utils.data.DataLoader(
        data.test, batch_size=64, shuffle=False, drop_last=False)
else:
    print('evaluation on train set of %s with model: %s' % (args.data, args.r))
    data = DataSet.create(args.data, test=False)
    data_loader = torch.utils.data.DataLoader(
        data.train, batch_size=64, shuffle=False, drop_last=False)

features, labels = extract_features(model, data_loader, print_freq=32, metric=None)
print('embedding dimension is:', len(features[0]))
num_class = len(set(labels))
print('number of classes is :', num_class)
print('compute the NMI index:', NMI(features, labels, n_cluster=num_class))

# print(len(features))
sim_mat = pairwise_similarity(features)
print(Recall_at_ks(sim_mat, query_ids=labels, gallery_ids=labels))
