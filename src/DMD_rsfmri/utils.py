import matplotlib.pyplot as plt
import numpy as np
import os


def createHankelMatrix2D(X, winSize):
    H = np.zeros((X.shape[0] * winSize, X.shape[1] - winSize + 1))
    for i in range(winSize, X.shape[1]):
        H[:, i-winSize] = X[:, i-winSize: i].flatten()
    return H


def summarize_dmd_modes(modes, lmd, psi):
    new_modes = []
    for i in range(lmd.shape[0]):
        new_modes.append(modes[:, i, np.where(lmd[i] > 0.95)].mean(axis=-1))
    new_modes = np.stack(new_modes, axis=1)
    new_modes /= np.linalg.norm(new_modes, axis=0)[np.newaxis]
    return new_modes[:, :, 0]


def sort_modes_on_freq(mode, psi, lmd):
    n = len(psi)
    idx = np.argsort(psi)[n//2:]
    return mode[:, idx], psi[idx], lmd[idx]


if __name__ == '__main__':
    savedir = "/Users/mturja/Downloads/HCP_PTN1200/ikeda_modes"
    netmat_dmd = np.loadtxt(os.path.join(savedir, f"netmats_dmd.txt"))
    for i in range(10, 30):
        mat = netmat_dmd[i].reshape(50, 3)[:, 0:1]
        mat = np.dot(mat, mat.T)
        plt.imshow(mat)
        plt.show()