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


class CenterLoss(nn.Module):
    def __init__(self, an_margin=0, ap_margin=0):
        super(CenterLoss, self).__init__()
        self.an_margin = an_margin
        self.ap_margin = ap_margin

    def forward(self, inputs, targets):
        n = inputs.size(0)
        num_dim = inputs.size(1)
        targets_ = list(set(targets.data))
        num_class = len(targets_)

        targets_ = Variable(torch.LongTensor(targets_))
        mask_ = targets.repeat(num_class, 1).eq(targets_.repeat(n, 1).t())
        centers = []
        inputs_list = []

        # calculate the centers for every class in one mini-batch
        for i, target in enumerate(targets_):
            mask_i = mask_[i].repeat(num_dim, 1).t()
            input_ = inputs[mask_i].resize(len(inputs[mask_i]) // num_dim, num_dim)
            centers.append(torch.mean(input_, 0))
            inputs_list.append(input_)

        centers = [centers[i].resize(1, num_dim) for i in range(len(centers))]
        centers = torch.cat(centers, 0)

        # compute centers loss here
        dist_ap, dist_an = [], []
        centers_dist = euclidean_dist(centers)

        for i, target in enumerate(targets_):
            dist_an.append(centers_dist[i][targets_ != target].min())
            center_diff = inputs_list[i] - centers[i]
            center_diff_norm = torch.cat([torch.norm(temp) for temp in center_diff])
            dist_ap.append(center_diff_norm.max())

        dist_an = torch.cat(dist_an)
        dist_ap = torch.cat(dist_ap)

        loss_an = torch.sum(dist_an[dist_an < self.an_margin])
        loss_ap = torch.sum(dist_ap[dist_ap > self.ap_margin])
        loss = loss_an + loss_ap
        return loss


def main():
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 3
    an_margin = 0.7
    ap_margin = 0.3
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    print('training data is ', x)
    print('initial parameters are ', w)
    inputs = x.mm(w)
    print('extracted feature is :', inputs)
    y_ = np.random.randint(num_class, size=data_size)
    targets = Variable(torch.from_numpy(y_))
    criterion = CenterLoss(ap_margin=ap_margin, an_margin=an_margin)
    loss = criterion(inputs, targets)
    print('loss is :', loss)

if __name__ == '__main__':
    main()
    print('Congratulations to you!')
#
#
