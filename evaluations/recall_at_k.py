# coding : utf-8
from __future__ import absolute_import
import heapq
import numpy as np
import random
from utils import to_numpy


def Recall_at_ks(sim_mat, query_ids=None, gallery_ids=None):
    """
    :param sim_mat:
    :param query_ids
    :param gallery_ids

    for the Deep Metric problem, following the evaluation table of Proxy NCA loss
    only compute the [R@1, R@2, R@4, R@8]

    fast computation via heapq

    """
    sim_mat = to_numpy(sim_mat)
    m, n = sim_mat.shape
    # Fill up default values
    gallery_ids = np.asarray(gallery_ids)
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    # Ensure numpy array
    if m > int(1e4):
        samples = list(range(m))
        random.shuffle(samples)
        samples = samples[:int(1e4)]
        sim_mat = sim_mat[samples, :]
        query_ids = [query_ids[k] for k in samples]
        m = int(1e4)
    else:
        query_ids = np.asarray(query_ids)

    # Sort and find correct matches
    # indice = np.argsort(sim_mat, axis=1)
    num_valid = np.zeros(4)
    for i in range(m):
        x = sim_mat[i]
        indice = heapq.nlargest(8, range(len(x)), x.take)    
        if query_ids[i] == gallery_ids[indice[0]]:
            num_valid += 1
        elif query_ids[i] == gallery_ids[indice[1]]:
            num_valid[1:] += 1
        elif query_ids[i] in gallery_ids[indice[1:4]]:
            num_valid[2:] += 1
        elif query_ids[i] in gallery_ids[indice[4:]]:
            num_valid[3] += 1
    return num_valid/float(m)


def main():
    import torch
    sim_mat = torch.rand(int(7e4), int(7*400))
    sim_mat = to_numpy(sim_mat)
    query_ids = int(1e4)*list(range(7))
    gallery_ids = int(1e3)*list(range(7))
    print(Recall_at_ks(sim_mat, query_ids, gallery_ids))

if __name__ == '__main__':
    main()