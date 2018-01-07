# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
import os
import sys
import torch.utils.data
from torch.backends import cudnn
import torch.optim as optim
from torch.autograd import Variable
import models
import losses
from utils import RandomIdentitySampler, mkdir_if_missing, logging
import DataSet
cudnn.benchmark = True


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-data', default='car', required=True,
                    help='path to dataset')
parser.add_argument('-loss', default='gaussian', required=True,
                    help='path to dataset')
parser.add_argument('-net', default='bn',
                    help='network used')
parser.add_argument('-r', default=None,
                    help='the path of the pre-trained model')
parser.add_argument('-start', default=0, type=int,
                    help='resume epoch')

parser.add_argument('-log_dir', default=None,
                    help='where the trained models save')

parser.add_argument('-BatchSize', '-b', default=128, type=int, metavar='N',
                    help='mini-batch size (1 = pure stochastic) Default: 256')
parser.add_argument('-num_instances', default=4, type=int, metavar='n',
                    help='')
parser.add_argument('-epochs', '-epochs', default=100, type=int, metavar='N',
                    help='epochs for training process')
parser.add_argument('-step', '-s', default=200, type=int, metavar='N',
                    help='number of epochs to save model')


# optimizer
parser.add_argument('-lr', type=float, default=1e-4,
                    help="learning rate of new parameters, for pretrained "
                         "parameters it is 10 times smaller than this")
parser.add_argument('--nThreads', '-j', default=4, type=int, metavar='N',
                    help='number of data loading threads (default: 2)')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)

args = parser.parse_args()

if args.log_dir is None:
    log_dir = os.path.join('checkpoints', args.loss)
else:
    log_dir = os.path.join('checkpoints', args.log_dir)
mkdir_if_missing(log_dir)
# write log
sys.stdout = logging.Logger(os.path.join(log_dir, 'log.txt'))


print('train on dataset %s' % args.data)
print('batchsize is: %d' % args.BatchSize)
print('num_instance is %d' % args.num_instances)
print('log dir is: %s' % args.log_dir)

if args.r is not None:
    model = torch.load(args.r)
else:
    model = models.create(args.net, Embed_dim=512)

    # load part of the model
    model_dict = model.state_dict()
    # print(model_dict)

    if args.net == 'bn':
        pretrained_dict = torch.load('pretrained_models/bn_inception-239d2248.pth')
    else:
        pretrained_dict = torch.load('pretrained_models/inception_v3_google-1a9a5a14.pth')

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # os.mkdir(log_dir)
    # torch.save(model, os.path.join(log_dir, 'model.pkl'))

model = model.cuda()
# print(model.parameters())

# criterion = CenterTripletLoss().cuda()
criterion = losses.create(args.loss).cuda()

# fine tune the model: the learning rate for pretrained parameter is 1/10
base_param_ids = set(map(id, model.Embed.parameters()))

#if model.classify:
#    base_param_ids = base_param_ids | set(map(id, model.logits_.parameters()))

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

data = DataSet.create(args.data, root=None, test=False)
train_loader = torch.utils.data.DataLoader(
    data.train, batch_size=args.BatchSize,
    sampler=RandomIdentitySampler(data.train, num_instances=args.num_instances),
    drop_last=True, num_workers=args.nThreads)


def adjust_learning_rate(opt_, epoch_, num_epochs):
    """Sets the learning rate to the initial LR decayed by 1000 at last 200 epochs"""
    if epoch_ > (num_epochs - args.step):
        lr = args.lr * \
             (0.001 ** ((epoch_ + args.step - num_epochs) / float(args.step)))
        for param_group in opt_.param_groups:
            param_group['lr'] = lr

for epoch in range(args.start, args.epochs):
    adjust_learning_rate(optimizer, epoch, args.epochs)
    running_loss = 0.0
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
        loss, inter_, dist_ap, dist_an = criterion(embed_feat, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
    # print(epoch)
    print('[epoch %05d]\t loss: %.7f \t prec: %.3f \t pos-dist: %.3f \tneg-dist: %.3f'
          % (epoch + 1,  running_loss, inter_, dist_ap, dist_an))
    if epoch % args.step == 0:
        torch.save(model, os.path.join(log_dir, '%d_model.pkl' % epoch))

    # if epoch == 2000:
    #     learn_rate /= 10
    #     optimizer = torch.optim.Adam(param_groups, lr=learn_rate,
    #                                  weight_decay=args.weight_decay)
    # if epoch == 3000:
    #     learn_rate /= 10
    #     optimizer = torch.optim.Adam(param_groups, lr=learn_rate,
    #                                  weight_decay=args.weight_decay)

torch.save(model, os.path.join(log_dir, '%d_model.pkl' % epoch))

print('Finished Training')
