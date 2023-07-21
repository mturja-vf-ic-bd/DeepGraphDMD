import numpy as np
from pydmd import DMD
from scipy.cluster.hierarchy import linkage
from utils import createHankelMatrix2D


class rsDMD:
    def __init__(self, recording, aug_win, dmd_win, t_res=0.72, svd_rank=0):
        self.X = recording
        self.h = aug_win
        self.win = dmd_win
        self.dt = t_res
        self.r = svd_rank

    def compute_dmd_modes(self):
        start_idx = 0
        modes = []
        while start_idx < self.X.shape[0]:
            X_aug = createHankelMatrix2D(self.X[:, start_idx: start_idx + self.win], self.h)
            dmd = DMD(svd_rank=self.r)
            dmd.fit(X_aug)
            modes.append(dmd.modes.reshape(self.X.shape[0], self.h, -1)[:, 0])
            start_idx += self.win
        modes = np.concatenate(modes, axis=1)
        return modes.T

    def cluster_dmd_modes(self, modes, method='average', metric='correlation'):
        Z = linkage(modes, method, metric)
        return Z


