from __future__ import absolute_import
import torch
import numpy as np
from utils import to_numpy


def Recall_at_1(distmat, query_ids=None, gallery_ids=None):
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
    num_valid_queries = 0
    for i in range(m):
        if query_ids[i] == gallery_ids[indices[i, 0]]:
            num_valid_queries += 1

    return num_valid_queries/float(m)


