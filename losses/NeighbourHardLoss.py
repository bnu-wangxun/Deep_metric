from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
# import numpy as np


class NeighbourHardLoss(nn.Module):
    def __init__(self, margin=0.05):
        super(NeighbourHardLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        eye_ = Variable(torch.eye(n)).cuda()
        eye_ = eye_.eq(1)
        pos_mask = mask - eye_

        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][pos_mask[i]].min())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        dist_ap = torch.mean(dist.masked_select(pos_mask)).data[0]
        dist_an = torch.mean(dist.masked_select(mask==0)).data[0]
        return loss, prec, dist_ap, dist_an


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

    print(NeighbourHardLoss(margin=0.1)(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')
