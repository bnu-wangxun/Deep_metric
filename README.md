# Deep Metric Learning
============================

## Pytorch Code for several deep metric learning papers:

- ["Lifted Structure Loss"](
https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Song_Deep_Metric_Learning_CVPR_2016_paper.pdf)

- Contrasstive Loss 

- Batch-All-Loss and Batch-Hard-Loss in ["In Defense of Triplet Loss in ReID"](https://arxiv.org/abs/1703.07737)

- New Loss for Master Graduation (in searching)

## Dataset
- [Car-196](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) 

   first 98 classes as train set and last 98 classes as test set
- [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200.html)

  first 98 classes as train set and last 98 classes as test set
  
- [Stanford-Online] 
  
  (Not Done yet)
  
## Prerequisites
- Computer with Linux or OSX
- [PyTorch](http://pytorch.org)
- For training, an NVIDIA GPU is strongly recommended for speed. CPU is supported but training is very slow.

## Reproducing LSUN experiments

**With BatchAllLoss:**

```bash
python train.py --dataset car --lr 1e-4
```
