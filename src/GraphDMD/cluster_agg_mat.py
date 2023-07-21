import os
import numpy as np
from src.CONSTANTS import CONSTANTS
import scipy.io
import torch
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--MIN_PSI", default=0.01, type=float)
parser.add_argument("--MAX_PSI", default=0.02, type=float)

args = parser.parse_args()

HOME = CONSTANTS.HOME
subject_ids = list(np.loadtxt(os.path.join(HOME, "filteredSubjectIds.txt")))
INPUT = f"/pine/scr/m/t/mturja/GraphDMD/rsfMRI_global_16_0.15"
agg_mat = []
DM_count = []
psi_arr = []
th_start = args.MIN_PSI
th_end = args.MAX_PSI

for i, s in enumerate(subject_ids):
    filename = os.path.join(INPUT, f"b_{i + 1}/sw_1.mat")
    if not os.path.exists(filename):
        break
    mat = scipy.io.loadmat(filename)
    phi = mat['Phi']
    lmd = np.abs(mat['Lambda'][:, 0])
    psi = mat['Psi'][:, 0]
    idx = np.where(np.logical_and(np.logical_and(lmd > 0.995, lmd < 1.005), np.logical_and(psi >= th_start, psi < th_end)))[0]
    phi_1 = phi[:, :, idx] + np.eye(phi.shape[0])[:, :, np.newaxis]
    phi_2 = -phi[:, :, idx] + np.eye(phi.shape[0])[:, :, np.newaxis]
    psi_arr.append(torch.from_numpy(psi[idx]))
    phi_real = torch.from_numpy(np.real(phi))
    phi_imag = torch.from_numpy(np.imag(phi))
    DM_count.append(len(idx))
    agg_mat.append(torch.from_numpy(np.real(phi_1)))
    agg_mat.append(torch.from_numpy(np.real(phi_2)))


def cluster_dms(agg_dm, n_cluster):
    from pyriemann.clustering import Kmeans
    kmeans = Kmeans(n_clusters=n_cluster, random_state=42, n_init=10)
    kmeans.fit(agg_dm)
    return kmeans.labels_


agg_mat = torch.cat(agg_mat, dim=2).permute(2, 0, 1)
psi_arr = torch.cat(psi_arr, dim=0)

n_cluster = 2
labels = cluster_dms(agg_mat.numpy(), n_cluster=n_cluster)
np.save("labels.npy", labels)
pos_cluster = 0

phi_algn = agg_mat[labels == pos_cluster]
start_ind = 0
for i, s in enumerate(subject_ids):
    file = os.path.join(HOME, f"gDMD_{th_start}_{th_end}_aligned", f"{int(s)}.txt")
    mat = phi_algn[start_ind:start_ind + DM_count[i]].mean(dim=0)
    mat -= torch.eye(mat.shape[0])
    mat = mat.numpy()
    pathlib.Path(os.path.join(HOME, f"gDMD_{th_start}_{th_end}_aligned")).mkdir(parents=True, exist_ok=True)
    np.savetxt(file, mat)
    print(f"Saved: {int(s)}")
    start_ind += DM_count[i]


