from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import pdb

def similarity(inputs_):
    # Compute similarity mat of deep feature
    # n = inputs_.size(0)
    sim = torch.matmul(inputs_, inputs_.t())
    return sim

class WeightLoss(nn.Module):
    def __init__(self, alpha=10, beta=2, margin=0.5, hard_mining=None, **kwargs):
        super(WeightLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.hard_mining = hard_mining

    def forward(self, inputs, targets):
        n = inputs.size(0)

        # Compute similarity matrix
        sim_mat = similarity(inputs)

        # print(sim_mat)
        targets = targets.cuda()

        # split the positive and negative pairs

        eyes_ = Variable(torch.eye(n, n)).cuda()
        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        neg_mask = eyes_.eq(eyes_) - pos_mask
        pos_mask = pos_mask - eyes_.eq(1)

        pos_sim = torch.masked_select(sim_mat, pos_mask)
        neg_sim = torch.masked_select(sim_mat, neg_mask)

        # 
        num_instances = len(pos_sim)//n + 1
        num_neg_instances = n - num_instances

        pos_sim = pos_sim.resize(len(pos_sim)//(num_instances-1), num_instances-1)
        neg_sim = neg_sim.resize(
            len(neg_sim) // num_neg_instances, num_neg_instances)

        base = 0.5

        loss = list()
        c = 0
        scale_value = self.margin

        for i, pos_pair_ in enumerate(pos_sim):
            # print(i)
            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_sim[i])[0]

            if self.hard_mining is not None:
                # print(self.alpha,  self.beta, self.margin)
                neg_pair = torch.masked_select(neg_pair_, neg_pair_ + 0.1 > pos_pair_[0])
                pos_pair = torch.masked_select(pos_pair_, pos_pair_ - 0.1 <  neg_pair_[-1])
                
                if len(neg_pair) < 1 or len(pos_pair) < 1:
                    c += 1
                    continue

                pos_loss = 2.0/self.beta * torch.log(1 + torch.sum(torch.exp(-self.beta * (pos_pair - base))))
                neg_loss = 2.0/self.alpha * torch.log(1 + torch.sum(torch.exp(self.alpha * (neg_pair - base))))
                loss.append(pos_loss + neg_loss)

            else:
                # print('hello world')
                neg_pair = neg_pair_
                pos_pair = pos_pair_
                pos_loss = 2.0/self.beta * torch.log(1 + torch.sum(torch.exp(-self.beta * (pos_pair - base))))
                neg_loss = 2.0/self.alpha * torch.log(1 + torch.sum(torch.exp(self.alpha * (neg_pair - base))))

                loss.append(pos_loss + neg_loss)

        
        for i in range(len(loss)):
            loss[i] = torch.unsqueeze(loss[i], dim = -1)
        loss = torch.sum(torch.cat(loss))/n
        prec = float(c)/n
        # pdb.set_trace()
        neg_d = torch.mean(neg_sim).data
        pos_d = torch.mean(pos_sim).data

        return loss, prec, pos_d, neg_d


def main():
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    # print(x)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = 8*list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(WeightLoss()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')


