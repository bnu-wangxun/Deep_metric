"""
python -i 002_label_processor.py \
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
"""
counter = 0
for line in path_list:
    one_hot_label = np.eye(len(label_list))[label_list.index(
        os.path.basename(line).split('_')[1].split('.')[0])]
    # one_hot_label = np.eye(len(label_list))[label_list.index(line.split('/')[-2])]
    one_hot_label = np.uint8(one_hot_label)
    path_label_dict[line] = one_hot_label
    counter += 1

np.save(os.path.join(config.save_path, 'path_label_dict.npy'), path_label_dict)
"""

"""
for itr in range(config.iter):
    # save the file inside of the meta/ folder
    f = open(os.path.join(save_path, 'path_label_list{0:03d}.txt'.format(itr)), 'w')
    for line in path_list[int((itr)*lenth/config.iter):int((itr+1)*lenth/config.iter)]:
        f.write(line + '\n')
    f.close()
"""