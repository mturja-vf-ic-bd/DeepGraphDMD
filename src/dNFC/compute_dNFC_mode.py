import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import argparse

from src.CONSTANTS import CONSTANTS
from src.GraphDMD.utils import get_subject_specific_modes, trial_similarity_mode
from src.SparseEdgeKoopman.utils import corr_coeff
from src.GraphDMD.cluster_gdmd_modes import summarize_dmd_modes, modified_k_means_single
from src.GraphDMD.analyse_cluster import compute_DT


def load_data(subjectID, trial, parcel):
    if trial == -1:
        fmri_signal = np.loadtxt(
            os.path.join(CONSTANTS.HOME,
                         "node_timeseries",
                         f"3T_HCP1200_MSMAll_d{parcel}_ts2",
                         f"{int(subjectID)}.txt")).T
    else:
        fmri_signal = np.loadtxt(
            os.path.join(CONSTANTS.HOME,
                         "node_timeseries",
                         f"3T_HCP1200_MSMAll_d{parcel}_ts2",
                         f"{int(subjectID)}.txt"))[(trial - 1) * 1200:trial * 1200].T
    return torch.from_numpy(fmri_signal)


def get_dNFC(bold_signal, window_size, stride):
    bld_sig_un = bold_signal.unfold(-1, window_size, stride).permute(1, 0, 2)
    pearsn_dNFC = corr_coeff(bld_sig_un)
    # pearsn_dNFC[pearsn_dNFC < 0] = 0
    ind = torch.triu_indices(pearsn_dNFC.shape[1], pearsn_dNFC.shape[1], 1)
    return pearsn_dNFC[:, ind[0], ind[1]]


def get_ICA(bold_signal, n_components):
    from sklearn.decomposition import FastICA
    ica = FastICA(n_components=n_components, max_iter=300)
    return ica.fit_transform(bold_signal)


def get_PCA(bold_signal, n_components):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    return pca.fit_transform(bold_signal)


def compute_component_based_score_across_trials(subject_id, parcel, sld_win, stride, n_cluster=3, type="pca"):
    X = []
    for trial in range(1, 5):
        bold = load_data(subject_id, trial, parcel)
        prsn = get_dNFC(bold, sld_win, stride)
        prsn = prsn.numpy().T
        prsn /= np.linalg.norm(prsn, axis=0, keepdims=True)
        if type == "ica":
            prsn = get_ICA(prsn, n_components=n_cluster)
        else:
            prsn = get_PCA(prsn, n_components=n_cluster)
        prsn /= np.linalg.norm(prsn, axis=0, keepdims=True)
        X.append(prsn)
    score = trial_similarity_mode(X)
    return score


def grassman_kernel(phi_1, phi_2):
    return np.linalg.norm(np.dot(phi_1.T, phi_2), ord='fro') ** 2


def get_similarity_score(subject_id, parcel, sld_win, stride, n_cluster):
    X = []
    for trial in range(1, 5):
        bold = load_data(subject_id, trial, parcel)
        prsn = get_dNFC(bold, sld_win, stride)
        prsn = prsn.numpy().T
        prsn /= np.linalg.norm(prsn, axis=0, keepdims=True)
        X.append(prsn)
    split = prsn.shape[1]
    X = np.concatenate(X, axis=1)
    modes, cluster_labels, sil_score, _ = summarize_dmd_modes(
        X, "skpmeans",
        cluster_range=np.arange(n_cluster, n_cluster + 1), OUTPUT_DIR="")
    # plt.figure(figsize=(50, 5))
    # plt.plot(cluster_labels)
    # plt.show()
    trial_modes = get_subject_specific_modes(X, cluster_labels, split, num_trial=0)
    score = trial_similarity_mode(trial_modes)
    # for trial in range(1, 5):
    #     l = cluster_labels[(trial - 1) * split: trial * split]
    #     DT, NT, FT, _ = compute_DT(l)
    #     plt.bar(np.arange(0, len(DT)), DT)
    #     plt.title(f"Trial: {trial}")
    #     plt.show()
    #     print(f"DT: {DT}, NT: {NT}, FT: {FT}")
    #     print(np.mean(DT))
    return score


def align_modes_across_trials(subject_id, parcel, sld_win, stride, n_cluster):
    mode_list = []
    for trial in range(1, 5):
        bold = load_data(subject_id, trial, parcel)
        prsn = get_dNFC(bold, sld_win, stride)
        prsn = prsn.numpy().T
        prsn /= np.linalg.norm(prsn, axis=0, keepdims=True)
        modes, cluster_labels, sil_score, _ = summarize_dmd_modes(
            prsn, "skpmeans",
            cluster_range=np.arange(n_cluster, n_cluster + 1),
            OUTPUT_DIR="")
        mode_list.append(modes)
    return mode_list



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_id", type=str)
    parser.add_argument("--parcel", type=int, default=50)
    parser.add_argument("--sld_win", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    subject_ids = np.loadtxt("/Users/mturja/PycharmProjects/KVAE/src/GraphDMD/filteredSubjectIds.txt")[0:100]
    score_ica = []
    for s in subject_ids:
        scr = compute_ICA_across_trials(int(s), args.parcel, args.sld_win, args.stride)
        score_ica.append(scr)
        print(f"subject_{s}:{scr}")
    plt.hist(score_ica)
    plt.show()
    print(f"ICA similarity score: {score_ica}")
    # score_v_cluster = []
    # cluster_sizes = np.arange(4, 5)
    # for n_cluster in cluster_sizes:
    #     score = get_similarity_score(
    #         args.subject_id, args.parcel,
    #         args.sld_win, args.stride, n_cluster)
    #     score_v_cluster.append(score)
    #     print(f"Similarity score: {score}")
    # plt.plot(cluster_sizes, score_v_cluster)
    # plt.show()
