import math
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HOME = f"/Users/mturja/Downloads/HCP_PTN1200"
for idx in range(0, 1003):
    mat1 = np.loadtxt(os.path.join(HOME, f"netmats/3T_HCP1200_MSMAll_d{50}_ts2/netmats1.txt"))[idx]
    mat1 = np.where(mat1 > np.percentile(mat1, q=50), mat1, 0)
    mat2 = np.loadtxt(os.path.join(HOME, f"netmats/3T_HCP1200_MSMAll_d{50}_ts2/netmats2.txt"))[idx]
    mat2 = np.where(mat2 > np.percentile(mat2, q=50), mat2, 0)
    dmd_mode = np.loadtxt(os.path.join(HOME, f"netmats/3T_HCP1200_MSMAll_d{50}_ts2/netmats3_2.txt"))[idx]
    dmd_mode = np.where(dmd_mode > np.percentile(dmd_mode, q=50), dmd_mode, 0)

    corr = stats.pearsonr(mat1, dmd_mode)[0]
    if math.isnan(corr):
        continue
    # assert corr > 0.3, f"Low correlation at idx: {idx} -> {corr}"
    print("Correlation:", corr)

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(131)
    ax.imshow(mat2.reshape(50, 50))
    ax.set_title("Partial Corr")
    ax = fig.add_subplot(132)
    ax.imshow(mat1.reshape(50, 50))
    ax.set_title("Pearson Corr")
    ax = fig.add_subplot(133)
    ax.imshow(dmd_mode.reshape(50, 50))
    ax.set_title(f"DMD mode: {corr:0.2f}")
    plt.tight_layout()
    plt.show()
