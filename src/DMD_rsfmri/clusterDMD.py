from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import numpy as np
import os
from matplotlib import pyplot as plt
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from src.CONSTANTS import CONSTANTS


def summarize_dmd_modes(X, dist_th, fname):
    Z = linkage(X.T, 'average', 'correlation')
    cluster_label = fcluster(Z, dist_th, criterion='distance')
    cnt = Counter(cluster_label)
    # sort_idx = np.argsort(cluster_label)
    th = len(cluster_label) / (2 * len(cnt))
    modes = np.zeros((50, 32))
    i = 0
    for k, v in cnt.items():
        if v > th:
            modes[:, i] = X[:, cluster_label == k].mean(axis=1)[0:50]
            i += 1
    modes = np.stack(modes, axis=1)
    np.save(fname, modes)
    return modes


# Load data
subjectIDs = np.loadtxt(os.path.join(CONSTANTS.HOME, "subjectIDs.txt"))
print(f"Total subjects:{len(subjectIDs)}")
edge_data = []
for s in tqdm(subjectIDs):
    edge_data = np.load(os.path.join(CONSTANTS.HOME, f"ikeda_modes/modes_{int(s)}_32_trial=0.npy"))
    X = edge_data.swapaxes(0, 1).reshape(50, -1)
    X = np.concatenate([X.real, X.imag], axis=0)
    X = (X - X.mean(axis=0)[np.newaxis]) / X.std(axis=0)[np.newaxis]
    Path(os.path.join(CONSTANTS.HOME, f"ikeda_modes_summary")).mkdir(parents=True, exist_ok=True)
    summarize_dmd_modes(X, os.path.join(CONSTANTS.HOME, f"ikeda_modes_summary/modes_{int(s)}_32_trial=0.npy"))




    # X_sorted = X[:, sort_idx]
    # corr = np.corrcoef(X_sorted.T)
    # plt.imshow(corr)
    # plt.show()
    # print(cluster_label[sort_idx])


