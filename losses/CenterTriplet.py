from __future__ import print_function, absolute_import

import torch
from torch import nn
from torch.autograd import Variable
# import numpy as np


def euclidean_dist(inputs_):
    # Compute pairwise distance, replace by the official when merged
    n = inputs_.size(0)
    dist = torch.pow(inputs_, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs_, inputs_.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def pair_euclidean_dist(inputs_x, inputs_y):
    n = inputs_x.size(0)
    m = inputs_y.size(0)
    xx = torch.pow(inputs_x, 2).sum(dim=1, keepdim=True).expand(n, m)
    yy = torch.pow(inputs_y, 2).sum(dim=1, keepdim=True).expand(m, n).t()
    dist = xx + yy
    dist.addmm_(1, -2, inputs_x, inputs_y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class CenterTripletLoss(nn.Module):
    def __init__(self):
        super(CenterTripletLoss, self).__init__()
        # self.margin = margin
        # self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        # inputs = nn.l2n
        n = inputs.size(0)
        num_dim = inputs.size(1)
        targets_ = list(set(targets.data))
        num_class = len(targets_)

        targets_ = Variable(torch.LongTensor(targets_)).cuda()
        mask_ = targets.repeat(num_class, 1).eq(targets_.repeat(n, 1).t())
        # print(mask_.size())
        _mask = Variable(torch.ByteTensor(num_class, n).fill_(1)).cuda() - mask_
        centers = []
        inputs_list = []

        for i, target in enumerate(targets_):
            mask_i = mask_[i].repeat(num_dim, 1).t()
            input_ = inputs[mask_i].resize(len(inputs[mask_i]) // num_dim, num_dim)
            centers.append(torch.mean(input_, 0))
            inputs_list.append(input_)

        centers = [centers[i].resize(1, num_dim) for i in range(len(centers))]
        centers = torch.cat(centers, 0)

        # compute centers loss here
        # dist_ap, dist_an = [], []
        centers_dist = pair_euclidean_dist(centers, inputs)
        # exp_dist = torch.exp(-centers_dist)
        neg_dist = centers_dist[_mask].resize(num_class-1, n)
        pos_dist = centers_dist[mask_]
        prec = (torch.min(neg_dist, 0)[0].data > 1.0*pos_dist.data).sum() * 1./n

        dist_an = torch.mean(neg_dist).data[0]
        dist_ap = torch.mean(pos_dist).data[0]

        # pos_dist = pos_dist - 0.8
        # neg_dist = 3.2 - neg_dist
        # print(torch.log(pos_dist))
        # print(torch.sum(torch.exp(centers_dist), 0))
        loss = torch.mean(pos_dist.clamp(min=0.15) -
                          torch.log(torch.sum(torch.exp(-neg_dist.clamp(max=0.6)), 0)))

        # print(loss)
        return loss, prec, dist_ap, dist_an

