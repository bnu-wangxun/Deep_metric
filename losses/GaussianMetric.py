from __future__ import print_function, absolute_import

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


def GaussDistribution(dist_list):
    """

    :param dist_list:
    :return:
    """
    mean_value = torch.mean(dist_list)
    diff = dist_list - mean_value
    std = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    return mean_value, std


class GaussianMetricLoss(nn.Module):
    def __init__(self):
        super(GaussianMetricLoss, self).__init__()

    # def compute(self):

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # num_dim = inputs.size(1)

        dist_mat = euclidean_dist(inputs)

        ones_ = Variable(torch.ones(n)).cuda() == 1
        eyes_ = Variable(torch.eye(n)).cuda() == 1

        mask_pos = targets.repeat(n, 1).eq(targets.repeat(n, 1).t()) - eyes_
        mask_neg = ones_ - targets.repeat(n, 1).eq(targets.repeat(n, 1).t())

        pos_dist = torch.masked_select(dist_mat, mask_pos)
        neg_dist = torch.masked_select(dist_mat, mask_neg)

        # the number of positive and negative pairs that be chosen
        selected_num = pos_dist.__len__()//5
        pos_dist = pos_dist.sort()[0][:selected_num]
        neg_dist = neg_dist.sort()[0][:selected_num]

        pos_point = torch.mean(pos_dist)
        neg_point = torch.mean(neg_dist)

        selected_pos = torch.masked_select(pos_dist, pos_dist > neg_point)
        pos_num = selected_pos.__len__()
        selected_neg = torch.masked_select(neg_dist, neg_dist < pos_point)
        neg_num = selected_neg.__len__()
        
        diff = neg_point - pos_point
        if pos_num > 0 and neg_num > 0:
            loss = torch.mean(selected_pos) - torch.mean(selected_neg)
        elif pos_num > 0:
            loss = torch.mean(selected_pos)
        elif neg_num > 0:
            loss = -torch.mean(selected_neg)
        else:
            loss = torch.clamp(torch.mean(pos_dist), max=0)
        return loss, diff.data[0], pos_num, neg_num
