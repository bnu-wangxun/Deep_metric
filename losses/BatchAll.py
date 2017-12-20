from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


def euclidean_dist(inputs_):
    # Compute pairwise distance, replace by the official when merged
    n = inputs_.size(0)
    dist = torch.pow(inputs_, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs_, inputs_.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class BatchAllLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(BatchAllLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist_mat = euclidean_dist(inputs)
        targets = targets.cuda()
        # split the positive and negative pairs
        eyes_ = Variable(torch.eye(n, n)).cuda()
        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        neg_mask = eyes_.eq(eyes_) - pos_mask
        pos_mask = pos_mask - eyes_.eq(1)

        pos_dist = torch.masked_select(dist_mat, pos_mask)
        neg_dist = torch.masked_select(dist_mat, neg_mask)

        num_instances = len(pos_dist)//n + 1
        num_neg_instances = n - num_instances

        pos_dist = pos_dist.resize(len(pos_dist)//(num_instances-1), num_instances-1)
        neg_dist = neg_dist.resize(
            len(neg_dist)//(num_neg_instances), num_neg_instances)

        #  clear way to compute the loss first
        loss = list()
        prec = list()
        for i, pos_pair in enumerate(pos_dist):

            neg_dist_ = neg_dist[i].repeat(num_instances - 1, 1)
            pos_dist_ = pos_pair.repeat(num_neg_instances, 1)
            pos_dist_ = pos_dist_.t()
            pos_dist_ = pos_dist_.resize(num_neg_instances*(num_instances - 1))
            neg_dist_ = neg_dist_.resize(num_neg_instances * (num_instances - 1))

            y = neg_dist_.data.new()
            y.resize_as_(neg_dist_.data)
            y.fill_(1)
            y = Variable(y)
            # loss.append(nn.MarginRankingLoss(margin=0.2)(pos_dist_, neg_dist_, y))
            loss.append(self.ranking_loss(neg_dist_, pos_dist_, y))
            prec.append((neg_dist_.data > pos_dist_.data).sum() * 1. / y.size(0))
        loss = torch.mean(torch.cat(loss))
        prec = np.mean(prec)
        neg_dist_mean = torch.mean(neg_dist).data[0]
        pos_dist_mean = torch.mean(pos_dist).data[0]
        return loss, prec, pos_dist_mean, neg_dist_mean


def main():
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    # print('training data is ', x)
    # print('initial parameters are ', w)
    inputs = x.mm(w)
    # print('extracted feature is :', inputs)

    # y_ = np.random.randint(num_class, size=data_size)
    y_ = 8*list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(BatchAllLoss(margin=0.2)(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')
