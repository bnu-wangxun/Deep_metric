from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from utils import to_numpy
import numpy as np


def NMI(X, ground_truth, n_cluster=3):
    X = to_numpy(X)
    print('x_type:', type(X))
    print('label_type:', type(ground_truth))
    kmeans = KMeans(n_clusters=n_cluster, random_state=0, max_iter=int(1e2)).fit(X)
    nmi = normalized_mutual_info_score(ground_truth, kmeans.labels_)
    return nmi


def main():
    label = [1, 2, 3]*2

    X = np.array([[1, 2], [1, 4], [1, 0],
                  [4, 2], [4, 4], [4, 0]])

    print(NMI(X, label))

if __name__ == '__main__':
    main()
