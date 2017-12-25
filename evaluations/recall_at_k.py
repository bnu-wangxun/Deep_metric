from __future__ import absolute_import
import torch
import numpy as np
from utils import to_numpy


def Recall_at_ks(distmat, query_ids=None, gallery_ids=None):
    """
    :param distmat:
    :param query_ids
    :param gallery_ids
    :return [R@1, R@2， R@4， R@8】

    for the Deep Metric problem, following the evaluation table of Proxy NCA loss
    only compute the [R@1, R@2, R@4, R@8]
    """
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    num_valid = np.zeros(4)
    for i in range(m):
        if query_ids[i] == gallery_ids[indices[i, 0]]:
            num_valid += 1
        elif query_ids[i] == gallery_ids[indices[i, 1]]:
            num_valid[1:] += 1
        elif query_ids[i] in gallery_ids[indices[i, 1:4]]:
            num_valid[2:] += 1
        elif query_ids[i] in gallery_ids[indices[i, 4:8]]:
            num_valid[3:] += 1
    return num_valid/float(m)


