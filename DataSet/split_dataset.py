import os
import shutil

root = 'CUB_200_2011'
image_folder = 'CUB_200_2011/images'
dirs = os.listdir(image_folder)

sets = ['train', 'test']

for x in sets:
    if not os.path.exists(os.path.join(root, x)):
        os.mkdir(os.path.join(root, x))

paths = [os.path.join(root, x) for x in sets]

train_idx = range(100)
test_idx = range(100, 200)

for i in range(200):
    img_path = os.path.join(image_folder, dirs[i])
    if i < 100:
        shutil.move(img_path, paths[0])
    else:
        shutil.move(img_path, paths[1])



