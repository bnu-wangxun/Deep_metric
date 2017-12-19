import torch
# import argparse
import torch.utils.data
from torch.backends import cudnn

from evaluations import extract_features, pairwise_distance
from evaluations import Recall_at_1
# from models import inception_v3
from DataSet import CUB200
# from utils import RandomIdentitySampler

cudnn.benchmark = True
test = False
# parser = argparse.ArgumentParser(description='PyTorch CUB200 Testing')
#
# parser.add_argument('--resume', type=str, default='adam_center_triplet_model.pkl', metavar='PATH')
#
# args = parser.parse_args()
# model_path = 'checkpoints/' + args.resume
# torch.cuda.set_device(8)
# model_path = 'checkpoints/histogram/model.pkl'
model_path = 'checkpoints/histogram/histogram400_model.pkl'
cudnn.benchmark = True

# model = inception_v3(dropout=0.5)
model = torch.load(model_path)
model = model.cuda()

if test:
    data = CUB200(root='DataSet/CUB_200_2011', train=False)
    data_loader = torch.utils.data.DataLoader(
        data.test, batch_size=64, shuffle=False, drop_last=False)
else:
    data = CUB200(root='DataSet/CUB_200_2011', test=False)
    data_loader = torch.utils.data.DataLoader(
        data.train, batch_size=64, shuffle=False, drop_last=False)
# sampler = RandomIdentitySampler(data.train, num_instances=4)

# for index, (_, id) in enumerate(data.train):
#     print(index, id)
#     break\
#
# for i, data in enumerate(data_loader, 0):
#     # get the inputs
#     inputs, labels = data
#     print('iteration is: %d' % i)

features, labels = extract_features(model, data_loader, print_freq=1, metric=None)

# print(len(features))
dist_mat = pairwise_distance(features)
print(Recall_at_1(dist_mat, query_ids=labels, gallery_ids=labels))
