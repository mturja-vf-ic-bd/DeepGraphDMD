import os.path
import numpy as np

from src.CONSTANTS import CONSTANTS
import argparse

from src.GraphDMD.cluster_gdmd_modes import modified_k_means_single
from src.GraphDMD.utils import trial_similarity_mode


if __name__ == '__main__':
    DIR_SUFFIX = CONSTANTS.GraphDMDDIR + "/rsfMRIfull_window=64_featdim=16_th=0.2_step=4"
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_id", type=str)
    parser.add_argument("--min_psi", type=float, default=0.06,
                        help="Minimum frequency for which DMD modes will be considered")
    parser.add_argument("--max_psi", type=float, default=0.09,
                        help="Maximum frequency for which DMD modes will be considered")
    args = parser.parse_args()

    trial_modes = []
    for trial in range(1, 5):
        root = DIR_SUFFIX + f"_trial={trial}_psi={args.min_psi}_{args.max_psi}"
        modes = np.load(os.path.join(root, args.subject_id, "modes.npy"))
        # modes = np.sqrt(modes[0:modes.shape[0]//2, :] ** 2 + modes[modes.shape[0]//2:, :]**2)
        # modes -= modes.mean(axis=0)[np.newaxis]
        # modes /= np.linalg.norm(modes, axis=0, keepdims=True)
        trial_modes.append(modes)
    trial_modes = np.concatenate(trial_modes, axis=1)
    trial_modes = modified_k_means_single(trial_modes)
    trial_modes = [trial_modes[:, i*3:(i+1)*3] for i in range(4)]
    score = trial_similarity_mode(trial_modes, th=0.1)
    print(f"Similarity score: {score}")
