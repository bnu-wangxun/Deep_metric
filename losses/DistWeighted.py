from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import pdb
# import numpy as np



def similarity(inputs_):
    # Compute similarity mat of deep feature
    # n = inputs_.size(0)
    sim = torch.matmul(inputs_, inputs_.t())
    return sim


def GaussDistribution(data):
    """
    :param data:
    :return:
    """
    mean_value = torch.mean(data)
    diff = data - mean_value
    std = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    return mean_value, std


class DistWeightedLoss(nn.Module):
    def __init__(self, alpha=50, beta=2, margin=0, **kwargs):
        super(DistWeightedLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        # self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute similarity matrixr®
        sim_mat = similarity(inputs)
        # print(sim_mat)
        targets = targets.cuda()
        # split the positive and negative pairs
        eyes_ = Variable(torch.eye(n, n)).cuda()
        # eyes_ = Variable(torch.eye(n, n))
        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        neg_mask = eyes_.eq(eyes_) - pos_mask
        pos_mask = pos_mask - eyes_.eq(1)

        pos_sim = torch.masked_select(sim_mat, pos_mask)
        neg_sim = torch.masked_select(sim_mat, neg_mask)

        num_instances = len(pos_sim)//n + 1
        num_neg_instances = n - num_instances

        pos_sim = pos_sim.resize(len(pos_sim)//(num_instances-1), num_instances-1)
        neg_sim = neg_sim.resize(
            len(neg_sim) // num_neg_instances, num_neg_instances)

        #  clear way to compute the loss first
        loss = list()
        c = 0
        base = 0.5
        for i, pos_pair in enumerate(pos_sim):
            # print(i)
            # pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair = torch.sort(neg_sim[i])[0]
            neg_mean, neg_std = GaussDistribution(neg_pair)
            prob = torch.exp(torch.pow(neg_pair - neg_mean, 2) / (2*torch.pow(neg_std, 2)))
            neg_index = torch.multinomial(prob, len(neg_pair)//10, replacement=False)
            neg_index_ = neg_index.data
            neg_pair = neg_pair[neg_index_]

            # print(neg_index)
            # import pdb; pdb.set_trace()
            
            
            pos_loss = 2.0/self.beta * torch.log(1 + torch.sum(torch.exp(-self.beta * (pos_pair - base))))
            neg_loss = 2.0/self.alpha * torch.log(1 + torch.sum(torch.exp(self.alpha * (neg_pair - base))))
            loss.append(pos_loss + neg_loss)

        loss = sum(loss)/n
        prec = float(c)/n
        # pdb.set_trace()
        neg_d = torch.mean(neg_sim).item()
        pos_d = torch.mean(pos_sim).item()

        return loss, prec, pos_d, neg_d


def main():
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = 8*list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(DistWeightedLoss(margin=0.1)(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')

