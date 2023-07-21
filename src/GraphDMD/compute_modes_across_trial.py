import argparse
import os.path

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from src.CONSTANTS import CONSTANTS

from src.GraphDMD.cluster_gdmd_modes import read_gdmd_results, cluster_dmd_modes
from src.GraphDMD.utils import get_subject_specific_modes, \
    trial_similarity_mode, compute_cluster_activation


def cluster_gdmd_modes_across_trials(
        dir_suffix,
        subject_id,
        psi_range,
        cluster_range,
        trial_num,
        OUTPUT_DIR):
    Phi = []
    Psi = []
    Lmd = []
    if trial_num == 0:
        for trial in range(1, 5):
            phi, psi, lmd, omega, b0 = read_gdmd_results(
                dir_suffix + f"_trial={trial}", subject_id
            )
            Phi += phi
            Psi += psi
            Lmd += lmd
        sub_bnd = [len(phi), 2 * len(phi), 3 * len(phi), 4 * len(phi)]
    else:
        phi, psi, lmd, omega, b0 = read_gdmd_results(
            dir_suffix + f"_trial={trial_num}", subject_id
        )
        Phi += phi
        Psi += psi
        Lmd += lmd
        sub_bnd = [len(phi)]
    Phi, modes, cluster_id, boundary, sub_bnd, score, bestK = cluster_dmd_modes(
        Phi=Phi, Psi=Psi, Lambda=Lmd, min_psi=psi_range[0], max_psi=psi_range[1],
        sub_bnd=sub_bnd,
        cluster_range=cluster_range, OUTPUT_DIR=OUTPUT_DIR)
    activation_map = []
    print("Subject boundary: ", sub_bnd)
    for i in range(1, len(sub_bnd)):
        trial_bnd = boundary[boundary.index(sub_bnd[i - 1]):boundary.index(sub_bnd[i]) + 1]
        activation_map.append(compute_cluster_activation(cluster_id, trial_bnd, bestK))
    activation_map = np.concatenate(activation_map, axis=1)
    return Phi, cluster_id, sub_bnd, activation_map, score, modes


def check_for_completion(OUTPUT_PATH):
    if not os.path.exists(os.path.join(OUTPUT_PATH, "activation_maps.npy")):
        return False
    if not os.path.exists(os.path.join(OUTPUT_PATH, "modes.npy")):
        return False
    return True


if __name__ == '__main__':
    DIR_SUFFIX = CONSTANTS.GraphDMDDIR + "/rsfMRIfull_window=64_featdim=16_th=0.2_step=4"
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_id", type=str)
    parser.add_argument("--min_K", type=int, default=3,
                        help="Minimum number of clusters of a range for cross-validation")
    parser.add_argument("--max_K", type=int, default=7,
                        help="Maximum number of clusters of a range for cross-validation")
    parser.add_argument("--min_psi", type=float, default=0.06,
                        help="Minimum frequency for which DMD modes will be considered")
    parser.add_argument("--max_psi", type=float, default=0.09,
                        help="Maximum frequency for which DMD modes will be considered")
    parser.add_argument("--trial", type=int, default=1,
                        help="Trial number")
    parser.add_argument("--force", type=bool, default=False)
    args = parser.parse_args()

    if args.trial == 0:
        OUTPUT_PATH = os.path.join(DIR_SUFFIX + f"_trial_results_psi={args.min_psi}_{args.max_psi}", args.subject_id)
        Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    else:
        OUTPUT_PATH = os.path.join(DIR_SUFFIX + f"_trial={args.trial}_psi={args.min_psi}_{args.max_psi}", args.subject_id)
        Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    if not check_for_completion(OUTPUT_PATH) or args.force:
        X, cluster_label, split, \
        activation_map, sil_score, trial_modes = cluster_gdmd_modes_across_trials(
            DIR_SUFFIX, args.subject_id,
            (args.min_psi, args.max_psi),
            np.arange(args.min_K, args.max_K), args.trial,
            OUTPUT_DIR=OUTPUT_PATH)
        modes = get_subject_specific_modes(X, cluster_label, split, args.trial)
        score = trial_similarity_mode(modes)
        print(f"Total data points: {len(cluster_label)}")
        # print(f"Mode similarity score: {score}")

        # store results
        np.save(os.path.join(OUTPUT_PATH, "activation_maps.npy"), activation_map)
        np.save(os.path.join(OUTPUT_PATH, "modes.npy"), trial_modes)
        np.save(os.path.join(OUTPUT_PATH, "Sil_score.npy"), sil_score)
        plt.plot(np.arange(args.min_K, args.max_K), sil_score)
        plt.savefig(os.path.join(OUTPUT_PATH, "Sil_score.png"))
    else:
        print(f"Result already exist for {args.subject_id}! Skipping!")
