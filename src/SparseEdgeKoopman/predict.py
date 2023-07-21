from pathlib import Path

from src.CONSTANTS import CONSTANTS

import os
import numpy as np
import torch
import argparse
import scipy.io
# from matplotlib import pyplot as plt
#
# from src.SparseEdgeKoopman.post_processing import compute_adj_lkis_sparse_edge
# from src.SparseEdgeKoopman.trainer import plot_set_of_multivarite_ts
# from src.SparseEdgeKoopman.utils import corr_coeff


def postprocess(model, subject_id, window, trial, stride):
    def get_latent(x):
        output = model(x)
        return output["z_gauss"]

    def read_subject(subject_id):
        parcel = 50
        fmri_signal = np.loadtxt(
            os.path.join(CONSTANTS.HOME, "node_timeseries",
                         f"3T_HCP1200_MSMAll_d{parcel}_ts2",
                         f"{int(subject_id)}.txt")).T
        fmri_signal = (fmri_signal - fmri_signal.mean(axis=-1)[:, np.newaxis]) \
                      / fmri_signal.std(axis=-1)[:, np.newaxis]
        return fmri_signal

    def write_latent(subject_id, z):
        parcel = 50
        write_path = os.path.join(CONSTANTS.HOME,
                         "latent", f"3T_HCP1200_MSMAll_d{parcel}_ts2")
        write_path_matlab = os.path.join(CONSTANTS.HOME,
                                  "latent_mat", f"3T_HCP1200_MSMAll_d{parcel}_ts2")
        if not os.path.exists(write_path):
            Path(write_path).mkdir(parents=True, exist_ok=False)
        if not os.path.exists(write_path_matlab):
            Path(write_path_matlab).mkdir(parents=True, exist_ok=False)
        np.save(os.path.join(write_path, f"{int(subject_id)}_trial={trial}.npy"), z)
        scipy.io.savemat(os.path.join(write_path_matlab, f"{int(subject_id)}_trial={trial}.mat"), {"X": z})

    def save_latent(subject_id):
        x = read_subject(subject_id)
        # x = torch.from_numpy(x)[np.newaxis].float()

        x = torch.from_numpy(x)[np.newaxis].float()[:, :, trial*1200:(trial + 1)*1200]
        x = x.unfold(-1, window, stride).permute(0, 2, 1, 3)
        print("Input shape: ", x.shape)
        z = get_latent(x)[0].detach().cpu().numpy()
        print("Output shape: ", z.shape)
        write_latent(subject_id, z)
        return z

    save_latent(subject_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Save latent embedding from a checkpoint")
    parser.add_argument("--ckpt", type=str, help="Checkpoint path")
    parser.add_argument("--subject_id", type=str, help="Subject id")
    parser.add_argument("--window", type=int, help="Sliding window size. Must match with training window")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--trial", type=int, default=0)
    args = parser.parse_args()
    print(args)
    model = torch.load(args.ckpt)
    postprocess(model, args.subject_id, args.window, args.trial, args.stride)
