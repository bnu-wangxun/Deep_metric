# coding=utf-8
import argparse
import os
import torch.utils.data
import torch.nn as nn
from torch.backends import cudnn
import torch.optim as optim
from torch.autograd import Variable
from models import inception_v3
from DataSet import Car196
from losses import *
from utils import RandomIdentitySampler

torch.cuda.set_device(8)
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch Car-196 Training')
parser.add_argument('--data', default='Car196', required=False,
                    help='path to dataset')
parser.add_argument('--nThreads', '-j', default=4, type=int, metavar='N',
                    help='number of data loading threads (default: 2)')
parser.add_argument('--save_path', default='checkpoints/Adam5',
                    help='where the trained models save')


parser.add_argument('--BatchSize', '-b', default=128, type=int, metavar='N',
                    help='mini-batch size (1 = pure stochastic) Default: 256')
parser.add_argument('--num_instances', default=4, type=int, metavar='n',
                    help='')

parser.add_argument('--epochs', '-epochs', default=100, type=int, metavar='N',
                    help='epochs for training process')
# optimizer
parser.add_argument('--lr', type=float, default=1e-4,
                    help="learning rate of new parameters, for pretrained "
                         "parameters it is 10 times smaller than this")
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)

args = parser.parse_args()

model = inception_v3(dropout=None, classify=False)

# load part of the model
model_dict = model.state_dict()
# print(model_dict)

pretrained_dict = torch.load('pretrained_models/inception_v3_google-1a9a5a14.pth')

# 将pretrained_dict里不属于model_dict的键剔除掉
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

# 更新现有的model_dict
model_dict.update(pretrained_dict)
# 加载我们真正需要的state_dict
model.load_state_dict(model_dict)
# os.mkdir(args.save_path)
torch.save(model, os.path.join(args.save_path ,'model.pkl'))
model = model.cuda()
# print(model.parameters())

# criterion = CenterTripletLoss().cuda()
criterion = GaussianMetricLoss().cuda()

# fine tune the model: the learning rate for pretrained parameter is 1/10
base_param_ids = set(map(id, model.Embed.parameters()))

if model.classify:
    base_param_ids = base_param_ids | set(map(id, model.logits_.parameters()))

base_params = [p for p in model.parameters() if
               id(p) in base_param_ids]

new_params = [p for p in model.parameters() if
              id(p) not in base_param_ids]
# print(new_params)
param_groups = [
            {'params': base_params, 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]

learn_rate = args.lr
optimizer = optim.Adam(param_groups, lr=learn_rate,
                       weight_decay=args.weight_decay)
#
# for g in optimizer.param_groups:
#     print(g.get('lr_mult', 1))

# Schedule learning rate
# def adjust_lr(epoch):
#     lr = args.lr if epoch <= 1200 else \
#         args.lr * (0.01 ** ((epoch - 1200) / 1800.0))
#     for g in optimizer.param_groups:
#         g['lr'] = lr * g.get('lr_mult', 1)

data = Car196(root='DataSet/Car196', test=False)
# data = CUB200(root='DataSet/' + args.data, test=False)
# train_loader = torch.utils.data.DataLoader(
#     data.train, batch_size=16,
#     sampler=RandomIdentitySampler(data.train, num_instances=2))
#
train_loader = torch.utils.data.DataLoader(
    data.train, batch_size=args.BatchSize,
    sampler=RandomIdentitySampler(data.train, num_instances=args.num_instances),
    drop_last=True, num_workers=args.nThreads)
#

for epoch in range(args.epochs):
    # adjust_lr(epoch)

    running_loss = 0.0
    # enumerate(sequence, [start=0])，i序号，data是数据
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        # break
        # wrap them in Variable
        inputs = Variable(inputs.cuda())
        labels = Variable(labels).cuda()
        # inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        embed_feat = model(inputs)

        # loss = criterion(embed_feat, labels)
        loss, inter_, dist_ap, dist_an = criterion(embed_feat, labels)  # 将output和labels使用叉熵计算损失
        # print(loss, prec, dist_an, dist_ap)
        # prec = 1.0
        # dist_an = 1.0
        # dist_ap = 1.0

        loss.backward()  # 反向传播
        optimizer.step()  # 用SGD更新参数
        running_loss += loss.data[0]  # loss本身为Variable类型，所以要使用data获取其Tensor，因为其为标量，所以取0
    # print(epoch)
    print('[epoch %d]\t loss: %.7f \t diff: %.3f \t pos-nums: %d \tneg-num: %d'
          % (epoch + 1,  running_loss, inter_, dist_an, dist_ap))
    if epoch % 200 == 0:
        torch.save(model, os.path.join(args.save_path ,'gaussian%d_model.pkl' % epoch))

    if epoch == 1600:
        # torch.save(model, os.path.join(args.save_path ,'gaussian%d_model.pkl' % epoch))
        learn_rate = learn_rate/5
        optimizer = torch.optim.Adam(param_groups, lr=learn_rate,
                                     weight_decay=args.weight_decay)

    if epoch == 2400:
        # torch.save(model, os.path.join(args.save_path ,'gaussian%d_model.pkl' % epoch))
        learn_rate = learn_rate/5
        optimizer = torch.optim.Adam(param_groups, lr=learn_rate,
                                     weight_decay=args.weight_decay)
    if epoch == 3000:
        # torch.save(model, os.path.join(args.save_path ,'gaussian%d_model.pkl' % epoch))
        learn_rate = learn_rate/5
        optimizer = torch.optim.Adam(param_groups, lr=learn_rate,
                                     weight_decay=args.weight_decay)

torch.save(model, os.path.join(args.save_path ,'gaussian%d_model.pkl' % epoch))
# torch.save(model.state_dict(), 'checkpoints/triplet_params.pkl')

print('Finished Training')
# trainer = Trainer(model, criterion, optimizer, train_loader)
# trainer.run(NUM_EPOCHS)
