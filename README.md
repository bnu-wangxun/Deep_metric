# Deep Metric Learning
============================

## Pytorch Code for several deep metric learning papers:

- ["Lifted Structure Loss"](
https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Song_Deep_Metric_Learning_CVPR_2016_paper.pdf)

- Contrasstive Loss 

- Batch-All-Loss and Batch-Hard-Loss in ["In Defense of Triplet Loss in ReID"](https://arxiv.org/abs/1703.07737)

- New Positive Mining Loss based on Fussy Clustering 

   [SOTA on standard metric learning Datasets]

## Dataset
- [Car-196](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) 

   first 98 classes as train set and last 98 classes as test set
- [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200.html)

  first 98 classes as train set and last 98 classes as test set
  
- [Stanford-Online] 
  
  for the experiments, we split 59,551 images of 11,318 classes for training and 60,502
images of 11,316 classes for testing
  
## Prerequisites
- Computer with Linux or OSX
- [PyTorch](http://pytorch.org)
  
 # NOTE！！！
  To exactly reproduce the result in my paper, please make sure to use the same version of pytorch with me: 0.2.3
  there are some problem for other version to load the pretrained model of inception-BN.
  
- For training, an NVIDIA GPU is strongly recommended for speed. CPU is supported but training may be slow.

## Reproducing Car-196 (or CUB-200-2011) experiments

**With our loss based on fussy clustering:**

```bash
sh run_train.sh
```

To reproduce other experiments, you can edit the run_train.sh file by yourself.
