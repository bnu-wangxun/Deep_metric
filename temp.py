from __future__ import print_function, absolute_import
import torch
from torch.autograd import Variable
import torch.optim as optim
import models

model = models.create('bn')

params = [p for p in model.parameters()]
param_groups = [
            {'params': params, 'lr_mult': 0.01}]

optimizer = optim.Adam(param_groups, lr=0.01)

#
# def adjust_learning_rate(opt_, epoch_, num_epochs):
#     """Sets the learning rate to the initial LR decayed by 1000 at last 200 epochs"""
#     if epoch_ > (num_epochs - 200):
#         lr = args.lr * (0.001 ** ((epoch_ + 200 - num_epochs) / 200.0))
for param_group in optimizer.param_groups:
    param_group['lr'] = 0.001
    print(param_group['lr'])