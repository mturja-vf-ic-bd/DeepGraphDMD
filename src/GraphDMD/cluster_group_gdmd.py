import os
import warnings
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from src.CONSTANTS import CONSTANTS
import argparse

import numpy as np
from src.GraphDMD.cluster_gdmd_modes import modified_k_means_single, summarize_dmd_modes


def load_group_dmd_modes(dir):
    root = dir
    index_dict = {}
    running_idx = 0
    modes = []
    act_maps = []
    for i in range(1, 841):
        path = os.path.join(root, f"b_{i}", "modes.npy")
        if os.path.exists(path):
            mode = np.load(path)
            act_map = np.load(os.path.join(root, f"b_{i}", "activation_maps.npy"))
            if len(act_map.shape) == 3:
                act_map = act_map.swapaxes(0, 1).reshape(act_map.shape[1], -1)
            elif act_map.shape[1] < 1120:
                act_map_new = np.zeros((act_map.shape[0], 1120))
                act_map_new[:, 0:act_map.shape[1]] = act_map
                act_map = act_map_new
            else:
                act_map = act_map[:, 0:1120]
            index_dict[f"b_{i}"] = {"start": running_idx, "end": running_idx + mode.shape[1]}
            running_idx += mode.shape[1]
            modes.append(mode)
            act_maps.append(act_map)
        else:
            warnings.warn(f"b_{i} doesn't exist! (Skipping...)")
    return np.concatenate(modes, axis=1), np.concatenate(act_maps, axis=0), index_dict


def align_subject_modes(modes, cluster_id, index_dict, act_maps, save_dir):
    max_cluster = max(cluster_id) + 1
    for sub_id, indices in index_dict.items():
        mode = modes[:, indices["start"]: indices["end"]]
        c = cluster_id[indices["start"]: indices["end"]]
        act_map = act_maps[indices["start"]: indices["end"]]
        aligned_modes = np.zeros((mode.shape[0], max_cluster))
        aligned_act_map = np.zeros((max_cluster, act_map.shape[1]))
        for id in range(max_cluster):
            if id in c:
                aligned_modes[:, id] = mode[:, c == id].mean(axis=1)
                # aligned_act_map[id] = act_map[c == id].mean(axis=0)
        np.save(os.path.join(save_dir, f"{sub_id}", "aligned_modes.npy"), aligned_modes)
        # np.save(os.path.join(save_dir, f"{sub_id}", "aligned_act_map.npy"), aligned_act_map)
        np.save(os.path.join(save_dir, f"{sub_id}", "group_cluster_id.npy"), c)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_psi", type=float)
    parser.add_argument("--max_psi", type=float)
    args = parser.parse_args()
    # DIR = "/pine/scr/m/t/mturja/GraphDMD/rsfMRIfull_window=64_featdim=16_th=0.2_step=4_trial_results_psi=0.06_0.09"
    DIR = os.path.join(CONSTANTS.GraphDMDDIR,
                       f"rsfMRIfull_window=64_featdim=16_th=0.2_step=4_trial_results_psi={args.min_psi}_{args.max_psi}")
    modes, act_maps, index_dict = load_group_dmd_modes(DIR)
    modes = modified_k_means_single(modes)
    group_modes, cluster_id, sil_score, bestK = summarize_dmd_modes(modes, "spkmeans", np.arange(2, 3), DIR)
    align_subject_modes(modes, cluster_id, index_dict, act_maps, DIR)

    # plt.plot(np.arange(3, 4), sil_score)
    # plt.tight_layout()
    # plt.savefig(os.path.join(DIR, "sil_score_plot.png"))
    np.save(os.path.join(DIR, f"group_modes_{args.min_psi}_{args.max_psi}.npy"), group_modes)

    # for i in range(group_modes.shape[1]):
    #     mode_real = modes[0:2500, i].reshape(50, 50)
    #     # mode_real = mode_real[idx, :]
    #     # mode_real = mode_real[:, idx]
    #     mode_imag = modes[2500:, i].reshape(50, 50)
    #     # mode_imag = mode_imag[idx, :]
    #     # mode_imag = mode_imag[:, idx]
    #     fig = plt.figure(figsize=(10, 5))
    #     ax = fig.add_subplot(121)
    #     ax.imshow(mode_real)
    #     ax.title.set_text("Real")
    #     ax = fig.add_subplot(122)
    #     ax.imshow(mode_imag)
    #     ax.title.set_text("Imag")
    #     plt.tight_layout()
    #     plt.show()
