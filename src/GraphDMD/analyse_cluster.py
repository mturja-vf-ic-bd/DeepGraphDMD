import os.path

import matplotlib.pyplot as plt
import numpy as np

from src.GraphDMD.cluster_gdmd_modes import summarize_dmd_modes


def cluster_activation_map(activation_map, n_cluster):
    """
    Graph DMD algorithm produces a set of network modes which can coexist
    in an overlapping manner.  The activation_map provides the activation of
    these graph modes at a particular time. The goal of this method is to cluster
    similar activation pattern which may mean macro states.

    :param activation_map: (n_cluster x T) numpy array where
    each row corresponds to the activation of cluster in time
    :param n_cluster: number of macro states
    :return: cluster_labels
    """

    activation_map /= activation_map.sum(axis=0)[np.newaxis]  # Normalize activation between 0 and 1
    modes, cluster_label, score = summarize_dmd_modes(
        activation_map, "spkmeans",
        np.arange(n_cluster, n_cluster + 1))
    return cluster_label


def smooth_activation_map(activation_map, w=1):
    for i in range(w, activation_map.shape[1] - w):
        s = 0
        for j in range(-w, w+1):
            s += activation_map[:, i + j]
        activation_map[:, i] = s / (2*w + 1)
    return activation_map


def compute_DT(state_vector):
    n_state = max(state_vector) + 1
    cur_state = state_vector[0]
    dt = 1
    dwell_vec = np.zeros(n_state)
    dwell_vec_cnt = np.zeros(n_state) + 1e-5
    FT = np.zeros(n_state)
    FT[cur_state] += 1
    task_switch_cnt = 0
    task_switch_bnd = []
    for i in range(1, len(state_vector)):
        if state_vector[i] == cur_state:
            # no state switching
            dt += 1
        else:
            # state_switching
            dwell_vec[cur_state] += dt
            dwell_vec_cnt[cur_state] += 1
            dt = 1
            cur_state = state_vector[i]
            task_switch_cnt += 1
            task_switch_bnd.append(i)
        FT[cur_state] += 1
    return dwell_vec / dwell_vec_cnt, task_switch_cnt, \
           FT / len(state_vector), task_switch_bnd


if __name__ == '__main__':
    DIR_SUFFIX = "/Users/mturja/rsfMRIfull_window=64_featdim=16_th=0.2_step=4_trial_results_K_2_20/b_287"
    ac_map = np.load(os.path.join(DIR_SUFFIX, "activation_maps.npy"))
    split = ac_map.shape[2]
    ac_map = np.concatenate([ac_map[0], ac_map[1], ac_map[2], ac_map[3]], axis=-1)

    ac_map = smooth_activation_map(ac_map, w=2)
    labels = cluster_activation_map(ac_map, 4)
    plt.figure(figsize=(50, 5))
    plt.plot(labels)
    plt.show()
    for trial in range(1, 5):
        l = labels[(trial - 1)*split: trial*split]
        DT, NT, FT, tb = compute_DT(l)
        plt.bar(np.arange(0, len(DT)), DT)
        plt.title(f"Trial: {trial}")
        plt.show()
        print(f"DT: {DT * 4}, NT: {NT}, FT: {FT}")
        print(np.mean(DT) * 4)
        plt.figure(figsize=(100, 20))
        plt.tight_layout()
        plt.imshow(ac_map[:, (trial - 1)*split: trial*split])
        plt.vlines(tb, ymin=0, ymax=18, colors='red')
        plt.tight_layout()
        plt.show()
        # print(Counter(l))
