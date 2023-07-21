import os
from collections import Counter

import scipy.io
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.cluster import KMeans

HOME = "/Users/mturja/PycharmProjects/KVAE/data/"


def process_matlab_cells_individual(dir_path, loadphi=True):
    directory = os.path.join(HOME, dir_path)
    Phi = []
    Psi = []
    Lambda = []
    Omega = []
    for b in range(1000):
        phi = []
        psi = []
        lmd = []
        omega = []
        for sw in range(0, 1000):
            filename = os.path.join(directory, f"b_{b+1}/sw_{sw+1}.mat")
            if not os.path.exists(filename):
                break
            mat = scipy.io.loadmat(filename)
            if loadphi:
                if len(mat['Phi'].shape) == 2:
                    mat['Phi'] = mat['Phi'][:, :, np.newaxis]
                phi.append(mat['Phi'])
            psi.append(mat['Psi'])
            lmd.append(mat['Lambda'])
            omega.append(mat['Omega'])
        if len(psi) == 0:
            break
        if loadphi:
            Phi.append(np.concatenate(phi, axis=-1))
        Psi.append(np.concatenate(psi, axis=0))
        Lambda.append(np.concatenate(lmd, axis=0))
        Omega.append(np.concatenate(omega, axis=0))
    if loadphi:
        return np.concatenate(Phi, axis=-1), np.concatenate(Psi, axis=0), np.concatenate(Lambda, axis=0), np.concatenate(Omega, axis=0)
    else:
        return np.concatenate(Psi, axis=0), np.concatenate(Lambda, axis=0), np.concatenate(Omega, axis=0)
    # return np.stack(Psi, axis=0), np.stack(Lambda, axis=0), np.stack(Omega, axis=0)


def process_matlab_cells_single(dir_path, id, loadphi=True):
    directory = dir_path
    Phi = None
    b = id
    phi = []
    psi = []
    lmd = []
    omega = []
    b0 = []
    ttX = []
    for sw in range(0, 1000):
        if type(id) is str:
            filename = os.path.join(directory, f"{id}/sw_{sw+1}.mat")
        else:
            filename = os.path.join(directory, f"b_{b+1}/sw_{sw+1}.mat")
        if not os.path.exists(filename):
            print(f"File {filename} not exist")
            break
        mat = scipy.io.loadmat(filename)
        if loadphi:
            if len(mat['Phi'].shape) == 2:
                mat['Phi'] = mat['Phi'][:, :, np.newaxis]
            phi.append(mat['Phi'])
        psi.append(mat['Psi'])
        lmd.append(mat['Lambda'])
        omega.append(mat['Omega'])
        b0.append(mat['b0'])
        # ttX.append(mat['tx'])
    if len(psi) == 0:
        return [], [], [], [], []
    if loadphi:
        Phi = np.concatenate(phi, axis=-1)
    Psi = np.concatenate(psi, axis=0)
    Lambda = np.concatenate(lmd, axis=0)
    Omega = np.concatenate(omega, axis=0)
    B0 = np.concatenate(b0, axis=0)
    # ttX = np.concatenate(ttX, axis=0)
    if loadphi:
        return Phi, Psi, Lambda, Omega, B0, ttX
    else:
        return Psi, Lambda, Omega, B0


def get_mixture_components(X, n_cluster):
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=n_cluster)
    gmm.fit(X)
    return gmm.means_


def plot_avg_dmd_modes(Phi, Psi, Lambda, cluster_length, means, stds, n_cluster):
    fig = plt.figure(figsize=(n_cluster*5, 40))
    # Phi = np.abs(np.real(Phi))
    plot_id = 1
    for i, m in enumerate(means):
        # ax = fig.add_subplot(len(means), 4, plot_id)
        mode, gran_modes = modified_k_means(Phi[:, :, np.logical_and((np.abs(Psi - m) < stds[i])[:, 0], np.abs(np.abs(Lambda)[:, 0] - 1) < 0.12)], n_cluster=n_cluster)
        # mode = np.where(mode > np.percentile(mode, q=80), mode, 0)
        # ax.imshow(mode)
        # plot_id += 1
        # sum_c = 0
        # for c in cluster_length[:-1]:
        #     sum_c += c
        #     ax.axhline(y=sum_c, xmin=0, xmax=268)
        #     ax.axvline(x=sum_c, ymin=0, ymax=268)
        # ax.set_title(f'Freq: {m:.3f}', fontsize=16)
        for g in gran_modes:
            ax = fig.add_subplot(len(means), n_cluster, plot_id)
            g = np.where(g > np.percentile(g, q=80), g, 0)
            ax.imshow(g)
            plot_id += 1
            ax.set_title(f'Freq: {m:.3f}', fontsize=16)
            sum_c = 0
            if cluster_length is not None:
                for c in cluster_length[:-1]:
                    sum_c += c
                    ax.axhline(y=sum_c, xmin=0, xmax=268)
                    ax.axvline(x=sum_c, ymin=0, ymax=268)
    plt.tight_layout()
    plt.show()


def plot_avg_dmd_modes_single(Phi, Psi, Lambda, means, stds, n_cluster):
    fig = plt.figure(figsize=(n_cluster*5, 8))
    plot_id = 1
    lmd_mean = np.mean(np.abs(Lambda))
    for i, m in enumerate(means):
        mode = modified_k_means_single(Phi[:, :, np.logical_and((np.abs(Psi - m) < stds[i])[:, 0], np.abs(Lambda)[:, 0] >= lmd_mean - 0.01)])
        mode = np.where(mode > np.percentile(mode, q=50), mode, 0)
        ax = fig.add_subplot(1, len(means), plot_id)
        plot_id += 1
        ax.imshow(mode)
        ax.set_title(f'Freq: {m:.3f}', fontsize=16)
    plt.tight_layout()
    plt.show()


def get_avg_phi_segment(Phi, Psi, Lambda, B0, min_psi, max_psi):
    lmd_mean = np.mean(np.abs(Lambda))
    filter = np.logical_and(np.logical_and(Psi[:, 0] < max_psi, min_psi <= Psi[:, 0]),
                                 np.abs(Lambda)[:, 0] >= lmd_mean - 0.01)
    mode = Phi[:, :, filter]
    # mode = mode * np.abs(B0[filter, 0])[np.newaxis, np.newaxis]
    avg_mode = modified_k_means_single(mode)
    return avg_mode


def validate_kmeans(Phi):
    from sklearn.cluster import KMeans
    scores = []
    for c in range(2, 10):
        kmeans = KMeans(c, random_state=0)
        kmeans.fit(Phi)
        scores.append(kmeans.inertia_)
    return scores


def modified_k_means(Phi, n_cluster):
    from sklearn.cluster import KMeans
    Phi = np.real(Phi)
    N = Phi.shape[0]
    Phi = Phi.reshape(-1, Phi.shape[2]).swapaxes(0, 1)
    Phi = np.concatenate([Phi, -Phi], axis=0)
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(Phi)
    means = kmeans.cluster_centers_
    center_sum = means.sum(axis=1)
    if center_sum[0] > center_sum[1]:
        Phi = Phi[kmeans.labels_ == 0]
        granular_means = cluster_modes(Phi, n_cluster=n_cluster)
        return means[0].reshape(N, N), granular_means
    else:
        Phi = Phi[kmeans.labels_ == 1]
        granular_means = cluster_modes(Phi, n_cluster=n_cluster)
        return means[1].reshape(N, N), granular_means


def modified_k_means_single(Phi):
    from sklearn.cluster import KMeans
    Phi = np.real(Phi)
    N = Phi.shape[0]
    Phi = Phi.reshape(-1, Phi.shape[2]).swapaxes(0, 1)
    Phi = np.concatenate([Phi, -Phi], axis=0)
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(Phi)
    means = kmeans.cluster_centers_
    center_sum = means.sum(axis=1)
    if center_sum[0] > center_sum[1]:
        return means[0].reshape(N, N)
    else:
        return means[1].reshape(N, N)


def plot_result(Psi, Lambda, Omega, n_cluster=8):
    freq = Psi
    amp = np.exp(np.real(Omega) * 2 * np.pi)
    fig = plt.figure(figsize=(30, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    freq_p = freq.flatten()
    Lambda_p = Lambda.flatten()
    freq_p = freq_p[abs(np.abs(Lambda_p) - 1) < 0.2]
    # freq_p = freq_p[freq_p > 1e-2]
    means = get_mixture_components(freq_p[:, np.newaxis], n_cluster=n_cluster)
    means = sorted(means[:, 0])
    means_diff = [means[i] - means[i - 1] for i in range(1, len(means))]
    print(means_diff)
    ax1.hist(freq_p, bins=50)
    for i in range(len(means)):
        x_axis = np.arange(means[i] - 0.05, means[i] + 0.05, 0.001)
        ax1.plot(x_axis, norm.pdf(x_axis, means[i], 0.01))
    ax1.set_xticks(np.arange(0, 0.5, 0.025))
    ax1.title.set_text("Frequency Histogram")

    ax2.hist(amp.flatten(), bins=50)
    # ax2.set_xticks(np.arange(0.8, 1.3, 0.1))
    ax2.title.set_text("Histogram of amplitude")
    plt.tight_layout()
    plt.show()


def cluster_modes(Phi, n_cluster):
    kmeans = KMeans(n_clusters=n_cluster, random_state=0)
    N = int(np.sqrt(Phi.shape[1]))
    kmeans.fit(Phi)
    means = kmeans.cluster_centers_
    return [means[i].reshape(N, N) for i in range(n_cluster)]


def get_subject_specific_modes(X, cluster_labels, split, num_trial):
    n_cluster = max(cluster_labels) + 1
    trial_modes = []
    if num_trial == 0:
        for trial in range(1, 5):
            if isinstance(split, list):
                start_idx = split[trial - 1]
                end_idx = split[trial]
            else:
                start_idx = (trial - 1) * split
                end_idx = trial * split
            x = X[:, start_idx:end_idx]
            c = cluster_labels[start_idx:end_idx]
            modes = np.zeros((x.shape[0], n_cluster))
            for i in range(n_cluster):
                if i in c:
                    modes[:, i] = x[:, c == i].mean(axis=1)
            modes /= (np.linalg.norm(modes, axis=0, keepdims=True) + 1e-5)
            trial_modes.append(modes)
    else:
        start_idx = split[0]
        end_idx = split[1]
        x = X[:, start_idx:end_idx]
        c = cluster_labels[start_idx:end_idx]
        modes = np.zeros((x.shape[0], n_cluster))
        for i in range(n_cluster):
            if i in c:
                modes[:, i] = x[:, c == i].mean(axis=1)
        modes /= (np.linalg.norm(modes, axis=0, keepdims=True) + 1e-5)
        trial_modes.append(modes)
    return trial_modes


def mode_similarity(phi_1, phi_2):
    diag_vals = np.diag(np.dot(phi_1.T, phi_2))
    return diag_vals[diag_vals != 0].mean()


def mode_similarity_aligned(phi_1, phi_2, th):
    phi_1 -= phi_1.mean(axis=0)[np.newaxis]
    phi_2 -= phi_2.mean(axis=0)[np.newaxis]
    phi_1 /= phi_1.std(axis=0)[np.newaxis]
    phi_2 /= phi_2.std(axis=0)[np.newaxis]
    sim_mat = np.abs(np.dot(phi_1.T, phi_2) / phi_1.shape[0])
    print(sim_mat)
    score = 0
    cnt = 0
    mapping = []
    while len(sim_mat) > 0 and sim_mat.sum() > 0:
        s = np.max(sim_mat)
        if s < th:
            break
        score += s
        ind = np.unravel_index(np.argmax(sim_mat, axis=None), sim_mat.shape)
        mapping.append(ind)
        for i in range(2):
            sim_mat = np.delete(sim_mat, ind[i], i)
        cnt += 1
    if cnt == 0:
        return 0
    print(mapping)
    return score / cnt


def align_modes(phi_1, phi_2):
    sim_mat = np.abs(np.dot(phi_1.T, phi_2) / phi_1.shape[0])
    s = np.max(sim_mat)


def trial_similarity_mode(mode_list, th=-1):
    score = 0
    for i in range(len(mode_list)):
        for j in range(i+1, len(mode_list)):
            s = mode_similarity_aligned(mode_list[i], mode_list[j], th)
            score += s
    return score / (len(mode_list) * (len(mode_list) - 1) // 2)


def compute_cluster_activation(cluster_id, boundary, max_cluster):
    n_cluster = max_cluster
    activation_map = np.zeros((n_cluster, len(boundary)-1))
    for i in range(1, len(boundary)):
        act_ids = Counter(cluster_id[boundary[i - 1]:boundary[i]])
        for k, v in act_ids.items():
            activation_map[k-1, i-1] = v
    return activation_map


def sort_network_based_on_clustering(network, n_cluster=5):
    """
    Group the nodes of the network so that the nodes
    in the same cluster appear sequentially. This function will be
    used just for visualization purpose like CirclePlot.
    :param n_cluster: number of clusters
    :return: sorted_network, ids
    """
    # import nibabel as nib
    # from sklearn.cluster import AgglomerativeClustering
    # gfull = nib.load("/Users/mturja/Desktop/Mnet1.pconn.nii").get_fdata()
    # conn = np.where(gfull > np.percentile(gfull, q=0.7), gfull, 0)
    # agc = AgglomerativeClustering(n_clusters=n_cluster, affinity='precomputed', connectivity=conn, linkage='average')
    # agc.fit(1-conn)
    # sort_idx = np.argsort(agc.labels_)
    # print("label counters: ", Counter(agc.labels_))
    # cnt = Counter(agc.labels_)
    sort_idx = np.array([0, 1, 2, 3, 15, 19, 23, 24,
                         13, 18, 22, 25, 32, 33, 35, 37, 30,
                         27, 28, 31, 34, 36, 38, 39, 40, 41, 43,
                         42, 44, 45, 46, 47, 48, 49,
                         4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 20, 21, 26, 29])
    labels = np.array([0] * 8 + [1] * 9 + [2] * 10 + [3] * 7 + [4] * 16)
    s = ""
    for l in labels:
        s += str(l)
        s += ", "
    print(s)
    conn_sorted = network[sort_idx, :]
    conn_sorted = conn_sorted[:, sort_idx]
    return conn_sorted, sort_idx, labels



