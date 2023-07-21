import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from collections import Counter
from matplotlib import pyplot as plt
import os, scipy.io
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import laplacian

from src.GraphDMD.clutering_algorithms import GMM, KMeansCV, SphericalKMeansCV, KMedoidCV_JensenShannon
from src.GraphDMD.utils import compute_cluster_activation

# DIR = "/pine/scr/m/t/mturja/GraphDMD/rsfMRIfull_window=64_featdim=16_th=0.2_step=4"
DIR = "/Users/mturja/rsfMRIfull_window=64_featdim=8_th=0.12_step=1"
OUTPUT_DIR = None


def find_sparse_cluster(X, kmeans):
    def get_idx(label_arr, label):
        idx = np.where(label_arr == label)[0]
        idx = np.array(sorted(idx, key=lambda x: x % (X.shape[1] // 2)))
        return idx

    centers = kmeans.cluster_centers_.T
    centers /= np.linalg.norm(centers, axis=0, keepdims=True)
    N = int(np.sqrt(centers.shape[0]))
    centers = centers.reshape((N, N, -1))
    centers[centers < 0] = 0
    c1 = laplacian(centers[:, :, 0], normed=True)
    c2 = laplacian(centers[:, :, 1], normed=True)
    eig_a = sorted(np.linalg.eigvals(c1))
    eig_b = sorted(np.linalg.eigvals(c2))
    if eig_a[1] < eig_b[1]:
        idx = get_idx(kmeans.labels_, 0)
    else:
        idx = get_idx(kmeans.labels_, 1)
    return X[:, idx]


def modified_k_means_single(X):
    """
    This routine fixes the sign of the eigen vectors (columns of X).
    Any eigen-decomposition based technique (like DMD) can generate
    both (eigenvector, eigenvalue) pairs: (\Phi, \Lambda) and (-\Phi, -\Lambda).
    This is an issue when applying these techniques in a window-based manner
    in a time-series data (like fMRI). This method fixes this inconsistency by the
    changing the sign to a particular polarity.

    Moreover, DMD based algorithms can produce complex conjugate pairs of eigenvectors:
    (A + jB, A - jB). This routine also changes the sign of the complex part of one of them.

    Example:
    Let's assume X = [[2, 1, -5, 6, 10, 1.4, -2.9, 0],
                     [-2, -1, 5, -6, 10, 1.4, -2.9, 0],
                     [2, 1, -5, 6, -10, -1.4, 2.9, 0],
                     [-2, -1, 5, -6, -10, -1.4, 2.9, 0]]

                print(modified_k_means_single(X.T).T)
                => [[2, 1, -5, 6, 10, 1.4, -2.9, 0],
                     [2, 1, -5, 6, 10, 1.4, -2.9, 0],
                     [2, 1, -5, 6, 10, 1.4, -2.9, 0],
                     [2, 1, -5, 6, 10, 1.4, -2.9, 0]]

    :param X: numpy array of shape (2*N, n_samples).
    The first half of the columns X[:N, i] is the real part and
    the last half is the imaginary part X[N:, i]
    :return: sign modified X


    """
    Phi = np.concatenate([X[0:X.shape[0]//2], -X[0:X.shape[0]//2]], axis=1)
    kmeans = KMeans(n_clusters=2, n_init=5)
    kmeans.fit(Phi.T)
    X = np.concatenate([Phi,  np.concatenate([X[X.shape[0]//2:], X[X.shape[0]//2:]], axis=1)], axis=0)
    X = find_sparse_cluster(X, kmeans)
    Phi = np.concatenate([X[X.shape[0] // 2:], -X[X.shape[0] // 2:]], axis=1)
    kmeans = KMeans(n_clusters=2, n_init=5)
    kmeans.fit(Phi.T)
    X = np.concatenate([np.concatenate([X[:X.shape[0] // 2], X[:X.shape[0] // 2]], axis=1), Phi], axis=0)
    X = find_sparse_cluster(X, kmeans)
    return X


def summarize_dmd_modes(X, clustering_algo="spkmeans", cluster_range=np.arange(5, 10), OUTPUT_DIR=None):
    score = None
    bestK = None
    if clustering_algo == "gmm":
        cluster_label, modes = GMM(X.T, cluster_range)
    elif clustering_algo =="kmeans":
        cluster_label, modes, scores = KMeansCV(X.T, cluster_range)
    elif clustering_algo == "kmedoids_jensen":
        cluster_label, modes, score = KMedoidCV_JensenShannon(X.T, cluster_range)
    else:
        cluster_label, modes, score, bestK = SphericalKMeansCV(X.T, cluster_range=cluster_range, OUTPUT_DIR=OUTPUT_DIR)
    return modes, cluster_label, score, bestK


def read_gdmd_results(dir_path, id):
    directory = dir_path
    b = id
    phi = []
    psi = []
    lmd = []
    omega = []
    b0 = []
    for sw in range(0, 1200):
        if type(id) is str:
            filename = os.path.join(directory, f"{id}/sw_{sw+1}.mat")
        else:
            filename = os.path.join(directory, f"b_{b+1}/sw_{sw+1}.mat")
        if not os.path.exists(filename):
            print(f"File {filename} not exist")
            break
        mat = scipy.io.loadmat(filename)
        if len(mat['Phi'].shape) == 2:
            mat['Phi'] = mat['Phi'][:, :, np.newaxis]
        phi.append(mat['Phi'])
        psi.append(mat['Psi'])
        lmd.append(mat['Lambda'])
        omega.append(mat['Omega'])
        b0.append(mat['b0'])
    return phi, psi, lmd, omega, b0


def read_gdmd_result_multi(dir_path, ids):
    Phi = []
    Psi = []
    Lmd = []
    boundary = []
    for id in ids:
        phi, psi, lmd, omega, b0 = read_gdmd_results(dir_path, id)
        Phi += phi
        Psi += psi
        Lmd += lmd
        boundary.append(len(Phi))

    return Phi, Psi, Lmd, boundary


def save_subject_modes(modes, subject_id):
    np.save(os.path.join(OUTPUT_DIR, "modes.npy"), modes)
    print(f"Saved: {subject_id}.npy")


def cluster_dmd_modes(
        Phi, Psi, Lambda, sub_bnd,
        max_psi=0.2, min_psi=0, cluster_range=np.arange(5, 10),
        OUTPUT_DIR=None):
    # Prepare DMD modes for clustering
    X = []
    boundary = [0]
    freq = []
    lmd_fil = []
    idx = 0
    sub_id = 0
    phi_cnt = 0
    sub_bnd_mod = [0]
    for phi, psi, lmd in zip(Phi, Psi, Lambda):
        idx += 1
        if idx > sub_bnd[sub_id]:
            sub_bnd_mod.append(phi_cnt + sub_bnd_mod[-1])
            phi_cnt = 0
            sub_id += 1

        sel_idx = np.logical_and(
            np.logical_and(
                np.logical_and(psi[:, 0] < max_psi, psi[:, 0] >= min_psi),
                lmd[:, 0].imag > 0), np.abs(lmd[:, 0]) > 0.95)
        x = phi[:, :, sel_idx]
        phi_cnt += x.shape[2]
        freq.append(psi[sel_idx])
        lmd_fil.append(lmd[sel_idx])
        X.append(x)
        boundary.append(x.shape[-1] + boundary[-1])

    Phi = np.concatenate(X, axis=-1)
    Phi = Phi.reshape(-1, Phi.shape[2])
    Phi = np.concatenate([np.real(Phi), np.imag(Phi)], axis=0)
    Phi /= np.linalg.norm(Phi, axis=0, keepdims=True)
    sub_bnd_mod.append(Phi.shape[1])
    print(f"Data shape: {Phi.shape}")

    # Clustering
    Phi = modified_k_means_single(Phi)
    modes, cluster_id, score, bestK = summarize_dmd_modes(
        Phi, cluster_range=cluster_range, OUTPUT_DIR=OUTPUT_DIR)
    return Phi, modes, cluster_id, boundary, sub_bnd_mod, score, bestK


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_id", type=str)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--min_K", type=int, default=12,
                        help="Minimum number of clusters of a range for cross-validation")
    parser.add_argument("--max_K", type=int, default=13,
                        help="Minimum number of clusters of a range for cross-validation")
    parser.add_argument("--min_psi", type=float, default=0)
    parser.add_argument("--max_psi", type=float, default=0.2,
                        help="Maximum frequency for which DMD modes will be considered")
    args = parser.parse_args()
    # Create output directory
    if args.trial:
        DIR = DIR + f"_trial={args.trial}"
    OUTPUT_DIR = os.path.join(DIR, f"gdmd_avg_modes_{args.min_psi}_{args.max_psi}", f"{args.subject_id}")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    Phi, Psi, Lambda, sub_bnd = read_gdmd_result_multi(dir_path=DIR, ids=[args.subject_id])
    Phi, modes, cluster_id, boundary, sub_bnd, sil_score, bestK = cluster_dmd_modes(
        Phi, Psi, Lambda, sub_bnd,
        max_psi=args.max_psi, min_psi=args.min_psi,
        cluster_range=np.arange(args.min_K, args.max_K),
        OUTPUT_DIR=OUTPUT_DIR)
    save_subject_modes(modes, subject_id=args.subject_id)
    activation_map = compute_cluster_activation(cluster_id, boundary, max_cluster=bestK)
    np.save(os.path.join(OUTPUT_DIR, "activation_map.npy"), activation_map)
    #
    # from d3blocks import D3Blocks
    # d3 = D3Blocks()
    for i in range(modes.shape[1]):
        df = pd.DataFrame(columns=["source", "target", "value"])
        mode_real = modes[0:2500, i].reshape(50, 50)
        mode_imag = modes[2500:, i].reshape(50, 50)
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121)
        ax.imshow(mode_real)
        ax.title.set_text("Real")
        ax = fig.add_subplot(122)
        ax.imshow(mode_imag)
        ax.title.set_text("Imag")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"modes_{i}.png"))
        # plt.show()
        # for src in range(50):
        #     for dst in range(0, 50):
        #         if mode_real[src, dst] > 0:
        #             df = df.append({"source": str(src), "target": str(dst), "weight": mode_real[src, dst]}, ignore_index=True)
        # d3.chord(df, filepath=f'chord_demo_{i}.html')
        # print(df.shape)
    print("Done!")


