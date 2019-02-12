"""
python -i datalist_processor.py \
--data_path=/home/siit/navi/data/input_data/CUB_200_2011/images \
--save_path=/home/siit/navi/data/input_data/CUB_200_2011/ \
--path_label False
"""

import os
import argparse
import numpy as np
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, dest='data_path', default='/home/siit/navi/data/input_data/mnist_png/')
parser.add_argument('--data_name', type=str, dest='data_name', default='danbooru')
parser.add_argument('--save_path', type=str, dest='save_path', default='/home/siit/navi/data/meta_data/mnist_png/')

parser.add_argument('--n_classes', type=int, dest='n_classes', default=34)
parser.add_argument('--path_label', type=bool, dest='path_label', default=False)
parser.add_argument('--iter', type=int, dest='iter', default=1)
config, unparsed = parser.parse_known_args() 



def file_list(path, extensions, sort=True):
    result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) 
    for f in filenames if os.path.splitext(f)[1] in extensions]
    if sort:
        result.sort() 
    return result



# make the save dir if it is not exists
save_path = config.save_path
if not os.path.exists(save_path):
    os.mkdir(save_path)

path_list = file_list(config.data_path, ('.jpg','.png'), True)
lenth = len(path_list)


# label_list = list('0123456789') # for mnist
# label_list = ['trainA', 'trainB'] # for cat dog
# label_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
label_list = os.listdir(config.data_path)
label_list.sort()

path_label_dict = {}

f = open(os.path.join(config.save_path, 'train.txt'), 'w')
for line in glob.glob(config.data_path + '/0*/*'):
    label_index = label_list.index(line.split('/')[-2]) 

    f.write('{} {}\n'.format(line, label_index))
    
f.close()

f = open(os.path.join(config.save_path, 'test.txt'), 'w')
for line in glob.glob(config.data_path + '/1*/*'):
    label_index = label_list.index(line.split('/')[-2]) 

    f.write('{} {}\n'.format(line, label_index))
    
f.close()
