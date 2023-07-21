import os
import pathlib

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from src.CONSTANTS import CONSTANTS
import numpy as np
from pydmd import DMD

from src.DMD_rsfmri.utils import createHankelMatrix2D, summarize_dmd_modes, sort_modes_on_freq
from src.dNFC.compute_dNFC_mode import get_dNFC

parcel = 50
n_sub = 1003
trial_length = 1200
trial = 0  # There are 4 trials (0 to 3)
dt = 0.72  # Temporal resolution
winsize = 16
svd_rank = 5
chunks = 8  # divide the whole timeseries
            # into smaller chunks where dmd will be computed
chunk_size = trial_length // chunks
savedir = "/Users/mturja/Downloads/HCP_PTN1200/ikeda_modes"
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)


# Load data
subjectIDs = np.loadtxt(os.path.join(CONSTANTS.HOME, "subjectIDs.txt"))[0:1]
print(f"Total subjects:{len(subjectIDs)}")
fmri_signal = np.zeros((min(n_sub, len(subjectIDs)), parcel, trial_length), dtype=float)

for i, s in enumerate(subjectIDs):
    fmri_signal[i] = np.loadtxt(
        os.path.join(CONSTANTS.HOME,
                     "node_timeseries",
                     f"3T_HCP1200_MSMAll_d{parcel}_ts2",
                     f"{int(s)}.txt"))[trial*trial_length:(trial+1) * trial_length].T
fmri_signal = (fmri_signal - fmri_signal.mean(axis=-1)[:, :, np.newaxis]) \
              / fmri_signal.std(axis=-1)[:, :, np.newaxis]
print("Data loaded")

# Computed dmd modes for each subject and save them
netmat_dmd = np.zeros((len(subjectIDs), 150))
for i, s in tqdm(enumerate(subjectIDs)):
    modes = []
    lmd = []
    psi = []
    for c in range(0, 1200-chunk_size+1, 4):
        X = fmri_signal[i, :, c:c+chunk_size]
        X_aug = createHankelMatrix2D(X, winsize)

        # Perform DMD
        dmd = DMD(svd_rank=svd_rank)
        dmd.fit(X_aug)
        mode = dmd.modes.reshape(parcel, winsize, -1)[:, 0]
        l = dmd.eigs
        f = np.imag(np.log(dmd.eigs))
        mode, f, l = sort_modes_on_freq(mode, f, l)
        psi.append(f)
        lmd.append(l)
        modes.append(mode)
        # print(f"mode shape-{i}:", modes[-1].shape)
    psi = np.stack(psi, axis=1)
    lmd = np.stack(lmd, axis=1)
    modes = np.stack(modes, axis=2)
    modes = summarize_dmd_modes(modes, lmd, psi)
    modes = np.real(modes)
    # net0 = np.dot(modes, modes.T)
    # plt.imshow(net0)
    # plt.show()
    # plt.hist(psi.flatten())
    # plt.show()
    netmat_dmd[i] = modes.flatten()
np.savetxt(os.path.join(savedir, f"netmats_dmd.txt"), netmat_dmd)


