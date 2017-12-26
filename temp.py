from __future__ import print_function, absolute_import
import torch
from torch.autograd import Variable
import models

pretrained_dict = torch.load('pretrain_models/bn_inception-239d2248.pth')
model = models.create('bn')

# load part of the model
model_dict = model.state_dict()
# print(model_dict)

pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

# 更新现有的model_dict
model_dict.update(pretrained_dict)
# 加载我们真正需要的state_dict
model.load_state_dict(model_dict)

input = torch.Tensor(1, 3, 224, 224)
input = Variable(input)
output = model(input)
# print(output)