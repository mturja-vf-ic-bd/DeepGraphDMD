import matplotlib.pyplot as plt

from src.dNFC.compute_dNFC_mode import load_data, get_dNFC, get_PCA
from src.CONSTANTS import CONSTANTS

import numpy as np
import os
from scipy import stats


def get_ICA(bold_signal, n_components):
    from sklearn.decomposition import FastICA
    ica = FastICA(n_components=n_components, max_iter=300)
    return ica.fit_transform(bold_signal)


def matricize(g, N=50):
    idu = np.triu_indices(N, k=1)
    mat = np.zeros((N, N))
    mat[idu] = g
    mat = mat + mat.T
    return mat


def compute_ica(s, n_comp, method="ica", trial=-1):
    data = load_data(s, trial, 50)
    dNFC = get_dNFC(data, 16, 4)
    if method == "ica":
        ica_comp = get_ICA(dNFC.T, n_components=n_comp)
    elif method == "pca":
        ica_comp = get_PCA(dNFC.T, n_components=n_comp)
    return ica_comp


N = 50
subject_ids = list(np.loadtxt(os.path.join(CONSTANTS.HOME, "subjectIDs.txt"), dtype=int))
netmat1 = np.loadtxt(os.path.join(CONSTANTS.HOME, f"netmats/3T_HCP1200_MSMAll_d{N}_ts2/netmats1.txt"))
n_comp = 2
method = "ica"
ica_comp = compute_ica(subject_ids[0], n_comp, method=method)
plt.figure(figsize=(10, 20))
netmat_ica = np.zeros_like(netmat1)
corr_list = []
for i, s in enumerate(subject_ids):
    max_c = 0
    max_mat = None
    for j in range(n_comp):
        mat = matricize(ica_comp[:, j], N)
        c = stats.pearsonr(mat.reshape(N*N, ), netmat1[i])[0]
        if abs(c) > abs(max_c):
            max_c = c
            max_mat = mat
    corr_list.append(abs(max_c))
    if max_c < 0:
        max_mat = -max_mat
    netmat_ica[i] = max_mat.reshape((N*N,))

print(f"Correlation stat for {method}: mu: {np.mean(corr_list)}, std: {np.std(corr_list)}")
np.savetxt(os.path.join(CONSTANTS.HOME, f"netmats/3T_HCP1200_MSMAll_d{N}_ts2/netmats_{method}_1.txt"), netmat_ica)
